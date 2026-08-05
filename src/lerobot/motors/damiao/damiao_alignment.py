#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Startup soft-alignment for OpenArm (Damiao) arms.

Port of the official openarm_teleop ``Control::AdjustPosition``
(src/controller/control.cpp): on startup, the arm is interpolated linearly
from its current pose to a fixed initial pose over ~2.2 s (220 steps x 10 ms)
using gains much softer than the teleop tracking gains. Running this on both
leader and follower at connect time means teleop starts with the two arms
already matched, without manual pose alignment and without the follower
jumping to the leader pose on the first tracking command.
"""

import logging
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..motors_bus import MotorCalibration

if TYPE_CHECKING:
    from .damiao import DamiaoMotorsBus

logger = logging.getLogger(__name__)

# Official INITIAL_POSITION (openarm_constants.hpp):
#   {0, 0, 0, pi/5, 0, 0, 0, 0} rad -> all joints 0, elbow (joint_4) raised
#   36 deg, gripper 0 (= closed, the calibration zero). lerobot's Damiao layer
#   works in degrees.
OPENARM_INITIAL_POSITION_DEG: dict[str, float] = {
    "joint_1": 0.0,
    "joint_2": 0.0,
    "joint_3": 0.0,
    "joint_4": 36.0,
    "joint_5": 0.0,
    "joint_6": 0.0,
    "joint_7": 0.0,
    "gripper": 0.0,
}

# Official AdjustPosition soft gains (kp_arm_temp/kd_arm_temp and
# kp_hand_temp/kd_hand_temp in control.cpp). Same MIT gain space as the teleop
# gains: positions are converted deg->rad inside _encode_mit_packet, the gains
# themselves pass through untouched (rad-based at the motor).
OPENARM_ALIGN_KP: dict[str, float] = {
    "joint_1": 50.0,
    "joint_2": 50.0,
    "joint_3": 50.0,
    "joint_4": 50.0,
    "joint_5": 10.0,
    "joint_6": 10.0,
    "joint_7": 10.0,
    "gripper": 10.0,
}
OPENARM_ALIGN_KD: dict[str, float] = {
    "joint_1": 1.2,
    "joint_2": 1.2,
    "joint_3": 1.2,
    "joint_4": 1.2,
    "joint_5": 0.3,
    "joint_6": 0.2,
    "joint_7": 0.3,
    "gripper": 0.5,
}

# Official cadence: nstep=220 steps of 10 ms.
_ALIGN_STEP_S = 0.01


def soft_move_to_position(
    bus,
    goal_pos_deg: dict[str, float],
    duration_s: float = 2.2,
    kp: dict[str, float] | None = None,
    kd: dict[str, float] | None = None,
    torque_ff_fn: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> None:
    """Softly interpolate motors from their current position to ``goal_pos_deg``.

    Linear interpolation (matching the official AdjustPosition), sending MIT
    position commands with soft gains every 10 ms. Blocks for ``duration_s``.

    Args:
        torque_ff_fn: optional callback ``pose_deg -> {motor: torque}`` giving a
            feed-forward torque to add to each step's MIT command, evaluated at
            the interpolated *target* pose. Used to inject gravity compensation
            during the ramp so a raised initial pose is actually reached and
            held (the soft gains alone sag under gravity at a lifted pose). When
            None the torque term stays 0.0 — identical to the official
            AdjustPosition and the previous behaviour.

    Raises:
        RuntimeError: if a motor in ``goal_pos_deg`` has not reported a
            position (commanding it from an unknown start pose is unsafe, and
            a silent state gap right after connect means bus trouble).
    """
    kp = kp if kp is not None else OPENARM_ALIGN_KP
    kd = kd if kd is not None else OPENARM_ALIGN_KD

    states = bus.sync_read_all_states()
    start_pos: dict[str, float] = {}
    for motor in goal_pos_deg:
        if motor not in bus.motors:
            continue
        position = states.get(motor, {}).get("position")
        if position is None:
            raise RuntimeError(
                f"soft_move_to_position: motor {motor!r} has no reported position after connect; "
                "refusing to command it from an unknown pose (check CAN wiring / motor power)."
            )
        start_pos[motor] = position

    n_steps = max(1, round(duration_s / _ALIGN_STEP_S))
    logger.info(
        f"Soft-aligning {len(start_pos)} motors to initial pose over {duration_s:.1f}s "
        f"(gravity feed-forward {'ON' if torque_ff_fn is not None else 'off'})..."
    )
    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        target = {motor: goal_pos_deg[motor] * alpha + start * (1.0 - alpha) for motor, start in start_pos.items()}
        ff = torque_ff_fn(target) if torque_ff_fn is not None else {}
        commands = {
            motor: (kp[motor], kd[motor], target[motor], 0.0, float(ff.get(motor, 0.0)))
            for motor in start_pos
        }
        bus._mit_control_batch(commands)
        time.sleep(_ALIGN_STEP_S)


def should_rezero_on_connect(
    calibration: dict[str, MotorCalibration] | None,
    override: bool | None = None,
) -> bool:
    """Decide whether ``connect()`` should burn a new motor-side zero (Damiao 0xFE).

    Historically the OpenArm classes re-burned the motor zero at EVERY connect,
    which forced the operator to hang the arm down identically each session and
    silently shifted the whole coordinate frame whenever they didn't (leader /
    follower mismatch, datasets with drifting frames). With a calibration that
    carries software homing offsets (see ``DamiaoMotorsBus._homing_offsets``)
    the zero is persistent and re-burning would DESTROY it.

    Decision, in order:
      - no calibration at all -> False (matches the legacy ``is_calibrated`` gate;
        a fresh ``calibrate()`` has just established the reference anyway);
      - explicit ``override`` (config ``rezero_on_connect``) -> honored as-is;
      - auto (``override is None``): legacy calibrations (all homing offsets
        zero) keep the historical per-session re-zero for backward compatibility;
        calibrations with any non-zero homing offset use the persistent zero and
        skip the burn.
    """
    if not calibration:
        return False
    if override is not None:
        return override
    return all(cal.homing_offset == 0 for cal in calibration.values())


def check_start_position(
    bus: "DamiaoMotorsBus",
    joint_limits: dict[str, tuple[float, float]],
    tolerance_deg: float | None,
    label: str = "OpenArm",
) -> None:
    """Plausibility gate before any startup motion when the per-session re-zero is skipped.

    With a persistent zero, a stale frame (motor zero reverted after a power
    cycle, calibration captured on a different arm, 0xFE burned out-of-band, or
    a single-turn wrap at boot) makes every position systematically wrong — and
    the startup alignment would then drive the arm to a wrong physical pose.
    This check reads all joint positions (logical frame) and refuses to proceed
    when any of them is outside ``joint_limits`` widened by ``tolerance_deg``:
    a resting arm with a valid frame always reads within its physical limits.
    It cannot catch small offsets (< tolerance), only gross frame corruption.

    Reads twice before failing so a transient CAN packet drop (stale cached
    state) does not abort a healthy startup. ``tolerance_deg=None`` disables the
    check entirely.

    Raises:
        RuntimeError: if any joint reads outside the widened limits.
    """
    if tolerance_deg is None:
        return

    bad: dict[str, float] = {}
    for attempt in range(2):
        states = bus.sync_read_all_states()
        bad = {}
        for motor, (lo, hi) in joint_limits.items():
            state = states.get(motor)
            if state is None:
                continue
            pos = state["position"]
            if not (lo - tolerance_deg <= pos <= hi + tolerance_deg):
                bad[motor] = pos
        if not bad:
            return
        if attempt == 0:
            time.sleep(0.05)

    details = ", ".join(f"{motor}={pos:.1f}deg" for motor, pos in bad.items())
    raise RuntimeError(
        f"{label}: startup position sanity check failed: {details} outside joint_limits "
        f"widened by {tolerance_deg:.0f} deg. The persistent zero looks stale (motor zero "
        "reverted after a power cycle, calibration from another arm, or an out-of-band "
        "0xFE zero write). Refusing to start (the alignment move would drive the arm to a "
        "wrong physical pose). Fix: physically rest the arm, re-run the calibration "
        "(arm hanging straight down, gripper closed); or set rezero_on_connect=true to "
        "fall back to the legacy per-session zero; or raise/disable "
        "start_position_tolerance_deg if the limits are intentionally narrow."
    )


def load_initial_pose_file(path: str) -> dict[str, float]:
    """Load a captured teleop start pose (degrees) from a YAML file.

    Accepts either a flat ``{joint_1: 12.0, ...}`` mapping or one nested under a
    top-level ``joints:`` key (the format written by
    ``scripts/capture_initial_pose.py``). Extra keys are ignored by the caller.

    Raises:
        FileNotFoundError: if ``path`` does not exist (with a hint to capture it).
    """
    import yaml

    resolved = os.path.expanduser(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"initial_pose_path not found: {resolved!r}. Capture it first with "
            "scripts/capture_initial_pose.sh <left|right>, or unset initial_pose_path "
            "to fall back to the default pose."
        )
    with open(resolved) as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict) and isinstance(data.get("joints"), dict):
        data = data["joints"]
    if not isinstance(data, dict):
        raise ValueError(f"initial_pose file {resolved!r} must be a mapping of joint -> degrees.")
    return {str(k): float(v) for k, v in data.items()}


def resolve_initial_pose(
    initial_pose_deg: dict[str, float] | None = None,
    initial_pose_path: str | None = None,
    joint_limits: dict[str, tuple[float, float]] | None = None,
    keep_gripper_zero: bool = True,
) -> dict[str, float]:
    """Resolve the startup alignment target, in priority order.

    ``initial_pose_deg`` (explicit override) > ``initial_pose_path`` (captured
    YAML) > ``OPENARM_INITIAL_POSITION_DEG`` (the official 36-deg-elbow default).
    Missing joints are back-filled from the default so the returned dict always
    covers the full arm. The result is clamped into ``joint_limits`` when given,
    and the gripper is forced to its calibration zero (closed) unless
    ``keep_gripper_zero`` is False — commanding a non-zero gripper hold on
    startup is what damaged a follower gripper in the field (FleetSeek
    exp_01KRN7HRZS9YTE0NNYEVW3TNDW).
    """
    if initial_pose_deg is not None:
        pose = dict(initial_pose_deg)
    elif initial_pose_path:
        pose = load_initial_pose_file(initial_pose_path)
    else:
        pose = dict(OPENARM_INITIAL_POSITION_DEG)

    merged = dict(OPENARM_INITIAL_POSITION_DEG)
    merged.update({k: float(v) for k, v in pose.items() if k in merged})

    if keep_gripper_zero and "gripper" in merged:
        merged["gripper"] = 0.0

    if joint_limits:
        for motor, (lo, hi) in joint_limits.items():
            if motor in merged:
                clamped = min(max(merged[motor], lo), hi)
                if abs(clamped - merged[motor]) > 1e-6:
                    logger.warning(
                        f"initial pose {motor}={merged[motor]:.1f}deg outside limits "
                        f"[{lo:.1f}, {hi:.1f}] -> clamped to {clamped:.1f}deg"
                    )
                merged[motor] = clamped

    return merged
