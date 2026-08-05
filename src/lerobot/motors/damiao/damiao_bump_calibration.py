#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bump-to-stop anchor for the OpenArm (Damiao) persistent software zero.

Port of the official ``openarm_can`` zero-position calibration
(``openarm_can/setup/openarm-can-zero-position-calibration``) to the lerobot
DamiaoMotorsBus MIT API, with the two known gaps of the official script fixed:

1. The official script measures per-joint travel-to-stop deltas but its
   ``move_to_precise_home`` is an unimplemented TODO, so it ends up burning the
   motor zero wherever the open-loop return moves happen to leave the arm
   (approximate hang-down, error accumulates over the relative moves).  This
   port instead computes the homing offset ANALYTICALLY from the raw reading at
   each stop contact (see :func:`compute_bump_homing_offsets`), so the zero is
   anchored to the mechanical stop itself, not to the end-of-sequence pose.
2. The official script "bumps" the gripper and then SKIPS it in the ideal-zero
   computation (``ideal[GRIPPER] skipped``), which caused a field breakage
   (leader/follower gripper zeros diverged -> teleop drove the gripper past its
   limit).  This port NEVER bumps or commands the gripper: its homing offset is
   pinned to 0.0 and the operator is directed to flash-zero it by hand at the
   fully-closed pose (``openarm_gripper_zero.py``) before running this
   calibration.

Offset formula
--------------

The lerobot Damiao layer defines ``logical = native - homing_offset`` on reads
and ``native = logical + homing_offset`` on MIT writes
(:class:`~lerobot.motors.damiao.damiao.DamiaoMotorsBus`).  The logical frame is
the URDF frame whose zero is the arm hanging straight down.

Each mechanical hard stop is a fixed physical configuration of the joint, so
its angle in the URDF frame is a known constant ``stop_angle_deg`` (the
matching end of the per-side URDF joint limits; verified against the official
script's ``ideal`` tables).  When the bump detects stop contact we read the
motor's NATIVE (pre-offset) position ``raw_at_stop``.  Requiring the logical
reading at the stop to equal the known stop angle:

    stop_angle = logical_at_stop = raw_at_stop - homing_offset
    =>  homing_offset = raw_at_stop - stop_angle

With this offset, the physical hang-down pose (URDF zero) reads logical 0 by
construction — no "move back to precise home" step is needed at all.

Safety
------

- The sequence actively sweeps every arm joint to a hard stop: the operator
  must clear the workspace and confirm before any torque is applied.
- Detection thresholds, per-unit stop angles, and the abort logic (silent
  motor, runaway travel) are conservative but WILL need per-unit tuning on the
  real arms (Thor): all of them are config-overridable
  (``bump_stop_angles_deg`` / ``bump_torque_thresholds_nm`` /
  ``bump_velocity_thresholds_deg_s`` on the OpenArm follower/leader configs).
- Values below are the official script's gains/thresholds converted to the
  lerobot degree-based API; they are NOT hardware-verified in this port yet.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .damiao import DamiaoMotorsBus

logger = logging.getLogger(__name__)

GRIPPER = "gripper"

# Official bump cadence: one MIT step every 5 ms, 0.2 deg per step (~40 deg/s
# command ramp; actual joint speed is lower because the gains are soft).
_BUMP_STEP_DEG = 0.2
_BUMP_STEP_PERIOD_S = 0.005
# Consecutive samples that must satisfy the stop condition before we accept the
# contact (the official script accepts a single sample; two in a row rejects
# single-sample torque/velocity glitches that would silently corrupt the
# offset).
_BUMP_HITS_REQUIRED = 2
# Consecutive CAN refresh misses before aborting (~= 40 * 5 ms = 0.2 s silent).
_BUMP_MAX_SILENT_STEPS = 40
# Extra command travel allowed beyond the joint's full range before aborting
# (a stop MUST have been hit within range + margin; more means the detection
# thresholds are wrong or the joint is free-spinning).
_BUMP_TRAVEL_MARGIN_DEG = 30.0
# Official inter-step pause.
_SEQ_PAUSE_S = 0.5
# Official soft-interpolation cadence (interp_time / 500 steps ~ 4 ms; we use
# 10 ms like damiao_alignment for lighter bus load, same total duration).
_MOVE_STEP_S = 0.01

# Hold gains while other joints are being bumped (official main(): arm kp
# [300, 300, 150, 150, 40, 40, 30], kd [2.5 x4, 0.8 x3]; gripper kp 10, kd 0.9
# holding its CURRENT position — never a fixed target).
_HOLD_KP = {
    "joint_1": 300.0,
    "joint_2": 300.0,
    "joint_3": 150.0,
    "joint_4": 150.0,
    "joint_5": 40.0,
    "joint_6": 40.0,
    "joint_7": 30.0,
    GRIPPER: 10.0,
}
_HOLD_KD = {
    "joint_1": 2.5,
    "joint_2": 2.5,
    "joint_3": 2.5,
    "joint_4": 2.5,
    "joint_5": 0.8,
    "joint_6": 0.8,
    "joint_7": 0.8,
    GRIPPER: 0.9,
}

# Stop-detection thresholds, ported from the official ``_hit_thresholds``:
#   joint_1 (J1): |vel| < 0.0125 rad/s (= 0.716 deg/s), |tau| > 5.0 Nm
#   joint_2..7  : |vel| < 0.1 rad/s    (= 5.73 deg/s),  |tau| > 2.0 Nm
# Torque thresholds must stay ABOVE the gravity/friction torque seen mid-travel
# at the sequence poses (or the bump false-triggers and the offset is wrong)
# and BELOW a level that could stress the hard stop. Tune per unit on Thor.
DEFAULT_BUMP_TORQUE_THRESHOLDS_NM: dict[str, float] = {
    "joint_1": 5.0,
    "joint_2": 2.0,
    "joint_3": 2.0,
    "joint_4": 2.0,
    "joint_5": 2.0,
    "joint_6": 2.0,
    "joint_7": 2.0,
}
DEFAULT_BUMP_VELOCITY_THRESHOLDS_DEG_S: dict[str, float] = {
    "joint_1": 0.716,
    "joint_2": 5.73,
    "joint_3": 5.73,
    "joint_4": 5.73,
    "joint_5": 5.73,
    "joint_6": 5.73,
    "joint_7": 5.73,
}

# Known mechanical stop angles in the LOGICAL (URDF hang-down-zero) frame, per
# side. These are the ends of the per-side URDF joint limits that the official
# per-side sequences bump into — cross-checked against the official ``ideal``
# tables (_run_right_sequence / _run_left_sequence):
#   right: J1 -80, J2 -10, J3 +90, J4 0, J5 +90, J6 +45, J7 +90 (deg)
#   left : J1 +80, J2 +10, J3 +90, J4 0, J5 +90, J6 +45, J7 +90 (deg)
# Physical tolerances make the true stop angle per unit differ slightly;
# override per unit via config ``bump_stop_angles_deg`` after measuring on Thor.
DEFAULT_BUMP_STOP_ANGLES_DEG: dict[str, dict[str, float]] = {
    "right": {
        "joint_1": -80.0,
        "joint_2": -10.0,
        "joint_3": 90.0,
        "joint_4": 0.0,
        "joint_5": 90.0,
        "joint_6": 45.0,
        "joint_7": 90.0,
    },
    "left": {
        "joint_1": 80.0,
        "joint_2": 10.0,
        "joint_3": 90.0,
        "joint_4": 0.0,
        "joint_5": 90.0,
        "joint_6": 45.0,
        "joint_7": 90.0,
    },
}


@dataclass(frozen=True)
class BumpStep:
    """Sweep ``motor`` in ``direction`` (+1/-1, logical frame) until its stop."""

    motor: str
    direction: float
    kp: float = 45.0  # official bump_to_limit default
    kd: float = 1.2


@dataclass(frozen=True)
class MoveStep:
    """Soft relative move of ``motor`` by ``delta_deg`` over ``duration_s``."""

    motor: str
    delta_deg: float
    duration_s: float = 2.0  # official interpolate default
    kp: float = 52.0
    kd: float = 1.5


# Per-side sequences: a faithful port of the official _run_right_sequence /
# _run_left_sequence ORDER and intermediate moves (they exist to avoid
# self-collision: e.g. J2 is nudged away from the body before the J3 bump, and
# the elbow is raised to horizontal before the wrist bumps), MINUS the gripper
# bump (never performed here, see module docstring). Move deltas are relative
# (frame-invariant), matching the official interpolate() semantics.
BUMP_SEQUENCES: dict[str, list[BumpStep | MoveStep]] = {
    "right": [
        BumpStep("joint_4", -1.0),
        MoveStep("joint_2", +5.0, duration_s=0.4),
        BumpStep("joint_3", +1.0),
        MoveStep("joint_3", -90.0, duration_s=1.0),
        MoveStep("joint_2", -5.0, duration_s=0.4),
        MoveStep("joint_4", +90.0),
        BumpStep("joint_5", +1.0),
        MoveStep("joint_5", -90.0),
        BumpStep("joint_6", +1.0),
        MoveStep("joint_6", -45.0),
        BumpStep("joint_7", +1.0),
        MoveStep("joint_7", -90.0),
        BumpStep("joint_2", -1.0),
        MoveStep("joint_2", +10.0, duration_s=0.9, kp=180.0, kd=2.0),
        BumpStep("joint_1", -1.0, kp=180.0, kd=2.1),
        MoveStep("joint_1", +80.0),
        MoveStep("joint_4", -90.0),
    ],
    "left": [
        BumpStep("joint_4", -1.0),
        MoveStep("joint_2", -5.0, duration_s=0.4),
        BumpStep("joint_3", +1.0),
        MoveStep("joint_3", -90.0, duration_s=1.0),
        MoveStep("joint_2", +5.0, duration_s=0.4),
        MoveStep("joint_4", +90.0),
        BumpStep("joint_5", +1.0),
        MoveStep("joint_5", -90.0),
        BumpStep("joint_6", +1.0),
        MoveStep("joint_6", -45.0),
        BumpStep("joint_7", +1.0),
        MoveStep("joint_7", -90.0),
        BumpStep("joint_2", +1.0),
        MoveStep("joint_2", -10.0, duration_s=0.9, kp=180.0, kd=2.0),
        BumpStep("joint_1", +1.0, kp=180.0, kd=2.1),
        MoveStep("joint_1", -80.0),
        MoveStep("joint_4", -90.0),
    ],
}


def compute_bump_homing_offsets(
    raw_at_stop_deg: dict[str, float],
    stop_angles_deg: dict[str, float],
) -> dict[str, float]:
    """Homing offsets from native readings at the mechanical stops.

    ``homing_offset = raw_at_stop - stop_angle`` (see module docstring for the
    derivation): with this offset installed, the logical reading at each stop
    equals its known URDF-frame angle, and the URDF hang-down pose reads
    logical zero for every joint.  Pure function so it can be validated
    offline with synthetic values.
    """
    missing = set(raw_at_stop_deg) - set(stop_angles_deg)
    if missing:
        raise ValueError(f"No known stop angle for joints: {sorted(missing)}")
    return {
        motor: float(raw) - float(stop_angles_deg[motor]) for motor, raw in raw_at_stop_deg.items()
    }


def _read_state(bus: "DamiaoMotorsBus", motor: str) -> dict | None:
    """One refresh cycle for ``motor``; None when it did not respond."""
    refreshed = bus._batch_refresh([motor])
    if motor not in refreshed:
        return None
    return bus._last_known_states[motor]


def _bump_to_stop(
    bus: "DamiaoMotorsBus",
    step: BumpStep,
    dq_th_deg_s: float,
    tau_th_nm: float,
    travel_limit_deg: float,
) -> float:
    """Sweep one joint into its mechanical stop; return the NATIVE angle there.

    Soft MIT position stepping (official ``bump_to_limit`` port): the command
    target advances by ``_BUMP_STEP_DEG`` every ``_BUMP_STEP_PERIOD_S`` while
    the joint follows with soft gains. Stop contact = ``_BUMP_HITS_REQUIRED``
    consecutive samples with |vel| below / |torque| above threshold AND the
    torque pressing in the sweep direction (the sign check is an addition over
    the official script; a real stop always reacts against the drive
    direction, so it can only reject false positives such as gravity load).

    Aborts (RuntimeError) when the motor stops responding on CAN or when the
    commanded travel exceeds ``travel_limit_deg`` without contact.
    """
    motor = step.motor
    state = _read_state(bus, motor)
    if state is None:
        raise RuntimeError(f"bump_to_stop({motor}): no CAN response before bump; aborting.")

    q_target = float(state["position"])  # logical frame
    q_start = q_target
    hits = 0
    silent = 0

    while True:
        q_target += step.direction * _BUMP_STEP_DEG
        if abs(q_target - q_start) > travel_limit_deg:
            raise RuntimeError(
                f"bump_to_stop({motor}): swept {abs(q_target - q_start):.1f} deg without stop "
                f"contact (limit {travel_limit_deg:.1f} deg). Detection thresholds too high, "
                "wrong side sequence, or mechanical problem. Motors disabled."
            )
        bus._mit_control_batch({motor: (step.kp, step.kd, q_target, 0.0, 0.0)})
        time.sleep(_BUMP_STEP_PERIOD_S)

        state = _read_state(bus, motor)
        if state is None:
            silent += 1
            if silent >= _BUMP_MAX_SILENT_STEPS:
                raise RuntimeError(
                    f"bump_to_stop({motor}): no CAN response for {_BUMP_MAX_SILENT_STEPS} "
                    "consecutive steps mid-bump; aborting (check power / wiring)."
                )
            continue
        silent = 0

        vel = float(state["velocity"])
        tau = float(state["torque"])
        pressing = (tau * step.direction) > 0
        if abs(vel) < dq_th_deg_s and abs(tau) > tau_th_nm and pressing:
            hits += 1
        else:
            hits = 0
        if hits >= _BUMP_HITS_REQUIRED:
            break

    # Release the position wind-up: re-command the target at the measured pose
    # so the joint stops pressing into the stop while we read / pause.
    q_stop_logical = float(state["position"])
    bus._mit_control_batch({motor: (step.kp, step.kd, q_stop_logical, 0.0, 0.0)})

    raw = bus.read_raw_positions([motor])
    if motor not in raw:
        raise RuntimeError(f"bump_to_stop({motor}): no raw position response at the stop; aborting.")
    logger.info(
        f"bump_to_stop({motor}): stop contact after {abs(q_stop_logical - q_start):.1f} deg "
        f"(native reading {raw[motor]:.2f} deg)"
    )
    return raw[motor]


def _soft_move_relative(bus: "DamiaoMotorsBus", step: MoveStep) -> None:
    """Relative soft move (official ``interpolate`` port), logical frame."""
    motor = step.motor
    state = _read_state(bus, motor)
    if state is None:
        raise RuntimeError(f"soft move({motor}): no CAN response; aborting.")
    q0 = float(state["position"])
    q1 = q0 + step.delta_deg
    n_steps = max(1, round(step.duration_s / _MOVE_STEP_S))
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        q = q0 + (q1 - q0) * alpha
        bus._mit_control_batch({motor: (step.kp, step.kd, q, 0.0, 0.0)})
        time.sleep(_MOVE_STEP_S)


def run_bump_to_stop_calibration(
    bus: "DamiaoMotorsBus",
    side: str,
    joint_limits: dict[str, tuple[float, float]],
    stop_angles_override: dict[str, float] | None = None,
    torque_thresholds_nm: dict[str, float] | None = None,
    velocity_thresholds_deg_s: dict[str, float] | None = None,
    label: str = "OpenArm",
    confirm: bool = True,
) -> dict[str, float]:
    """Full bump-to-stop calibration for one arm; returns homing offsets (deg).

    Runs the per-side official sequence (minus the gripper), reading the NATIVE
    position at every stop contact and converting it to a homing offset via
    :func:`compute_bump_homing_offsets`.  The gripper is pinned to offset 0.0:
    its zero MUST have been flash-burned at the fully-closed pose beforehand
    (``openarm_gripper_zero.py``) — this function never commands the gripper
    except to softly hold its current position.

    The arm may start from any pose within its workspace (readings at the
    stops are absolute; nothing depends on the start pose), but it MUST be
    free to sweep every joint to its limits.

    Motors are left with torque DISABLED on return and on any error.
    """
    if side not in BUMP_SEQUENCES:
        raise ValueError(f"side must be one of {sorted(BUMP_SEQUENCES)}, got {side!r}")

    stop_angles = dict(DEFAULT_BUMP_STOP_ANGLES_DEG[side])
    if stop_angles_override:
        stop_angles.update({k: float(v) for k, v in stop_angles_override.items()})
    tau_th = dict(DEFAULT_BUMP_TORQUE_THRESHOLDS_NM)
    if torque_thresholds_nm:
        tau_th.update({k: float(v) for k, v in torque_thresholds_nm.items()})
    dq_th = dict(DEFAULT_BUMP_VELOCITY_THRESHOLDS_DEG_S)
    if velocity_thresholds_deg_s:
        dq_th.update({k: float(v) for k, v in velocity_thresholds_deg_s.items()})

    sequence = BUMP_SEQUENCES[side]
    arm_motors = [m for m in bus.motors if m != GRIPPER]
    seq_motors = {s.motor for s in sequence}
    if seq_motors != set(arm_motors):
        raise ValueError(
            f"bump sequence joints {sorted(seq_motors)} do not match arm motors {sorted(arm_motors)}"
        )

    if confirm:
        print(
            f"\n{label}: BUMP-TO-STOP calibration ({side} arm sequence)\n"
            "  *** CLEAR THE WORKSPACE — the arm will actively sweep every joint to its\n"
            "  *** mechanical hard stops (shoulder, elbow, wrist). Keep hands clear and\n"
            "  *** keep the power switch / Ctrl-C within reach for the whole run.\n"
            "  - The gripper is NOT bumped. Its motor zero must already be flash-burned\n"
            "    at the fully-closed pose (scripts/openarm_gripper_zero.py); its homing\n"
            "    offset is pinned to 0.\n"
            "  - The arm can start from any pose, but every joint must be free to reach\n"
            "    its limits (no table/objects inside the sweep volume).\n"
            f"  - Confirm this is really the {side.upper()} arm: the {side} sequence on the\n"
            "    wrong arm drives joints toward the WRONG stops.\n"
        )
        input("Press ENTER to start the bump-to-stop sweep (Ctrl-C to abort)... ")

    raw_at_stop: dict[str, float] = {}
    try:
        # Hold every motor (gripper included) at its current position with the
        # official hold gains, so unbumped joints don't sag while others sweep.
        bus.enable_torque()
        states = bus.sync_read_all_states()
        hold_cmds = {}
        for motor in bus.motors:
            pos = states.get(motor, {}).get("position")
            if pos is None:
                raise RuntimeError(
                    f"{label}: motor {motor!r} reported no position before calibration; aborting."
                )
            hold_cmds[motor] = (_HOLD_KP[motor], _HOLD_KD[motor], float(pos), 0.0, 0.0)
        bus._mit_control_batch(hold_cmds)
        time.sleep(0.1)

        for step in sequence:
            if isinstance(step, BumpStep):
                lo, hi = joint_limits[step.motor]
                travel_limit = (hi - lo) + _BUMP_TRAVEL_MARGIN_DEG
                raw_at_stop[step.motor] = _bump_to_stop(
                    bus, step, dq_th[step.motor], tau_th[step.motor], travel_limit
                )
            else:
                _soft_move_relative(bus, step)
            time.sleep(_SEQ_PAUSE_S)

        offsets = compute_bump_homing_offsets(raw_at_stop, stop_angles)
        offsets[GRIPPER] = 0.0  # pinned: hand-zeroed at closed, never bumped

        # Sanity report: the sequence ends near hang-down, so the logical
        # readings (native - offset) should all be near 0 now. Warn only —
        # the return moves are open-loop and a few degrees of error is normal.
        raw_now = bus.read_raw_positions(arm_motors)
        for motor in arm_motors:
            if motor in raw_now:
                logical_now = raw_now[motor] - offsets[motor]
                if abs(logical_now) > 10.0:
                    logger.warning(
                        f"{label}: {motor} reads {logical_now:.1f} deg from hang-down after the "
                        "sequence (expected ~0). If the arm visibly IS hanging straight down, a "
                        "bump likely false-triggered or the per-unit stop angle is off — verify "
                        "before trusting this calibration (see the Thor runbook)."
                    )
        return offsets
    finally:
        bus.disable_torque()
