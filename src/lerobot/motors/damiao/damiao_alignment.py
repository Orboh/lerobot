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
import time

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
) -> None:
    """Softly interpolate motors from their current position to ``goal_pos_deg``.

    Linear interpolation (matching the official AdjustPosition), sending MIT
    position commands with soft gains every 10 ms. Blocks for ``duration_s``.

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
    logger.info(f"Soft-aligning {len(start_pos)} motors to initial pose over {duration_s:.1f}s...")
    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        commands = {
            motor: (kp[motor], kd[motor], goal_pos_deg[motor] * alpha + start * (1.0 - alpha), 0.0, 0.0)
            for motor, start in start_pos.items()
        }
        bus._mit_control_batch(commands)
        time.sleep(_ALIGN_STEP_S)
