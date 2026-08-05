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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

LEFT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    # URDF (openarm_v10_bimanual) の物理可動域に合わせた値（度）
    "joint_1": (-200.0, 80.0),
    "joint_2": (-190.0, 10.0),
    "joint_3": (-90.0, 90.0),
    "joint_4": (0.0, 140.0),
    "joint_5": (-90.0, 90.0),
    "joint_6": (-45.0, 45.0),
    "joint_7": (-90.0, 90.0),
    "gripper": (-65.0, 0.0),
}

RIGHT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    # URDF (openarm_v10_bimanual) の物理可動域に合わせた値（度）。左の符号反転
    "joint_1": (-80.0, 200.0),
    "joint_2": (-10.0, 190.0),
    "joint_3": (-90.0, 90.0),
    "joint_4": (0.0, 140.0),
    "joint_5": (-90.0, 90.0),
    "joint_6": (-45.0, 45.0),
    "joint_7": (-90.0, 90.0),
    "gripper": (-65.0, 0.0),
}


@dataclass
class OpenArmFollowerConfigBase:
    """Base configuration for the OpenArms follower robot with Damiao motors."""

    # CAN interfaces - one per arm
    # arm CAN interface (e.g., "can1")
    # Linux: "can0", "can1", etc.
    port: str

    # side of the arm: "left" or "right". If "None" default values will be used
    side: str | None = None

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
    can_interface: str = "socketcan"

    # CAN FD settings (OpenArms uses CAN FD by default)
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # Nominal bitrate (1 Mbps)
    can_data_bitrate: int = 5000000  # Data bitrate for CAN FD (5 Mbps)

    # Whether to disable torque when disconnecting
    disable_torque_on_disconnect: bool = True

    # When True, expose `.vel` and `.torque` per motor in observation features.
    # Default False for compatibility with the position-only openarm_mini teleoperator.
    use_velocity_and_torque: bool = False

    # Safety limit for relative target positions
    # Set to a positive scalar for all motors, or a dict mapping motor names to limits
    max_relative_target: float | dict[str, float] | None = None

    # --- Calibration anchor ----------------------------------------------------
    # How calibrate() establishes the persistent software zero:
    #   "hang_down" (default): operator hangs the arm straight down and the raw
    #     readings at that pose become the homing offsets (accuracy limited by
    #     how reproducibly a human can hang the arm).
    #   "bump_to_stop": the arm actively sweeps each ARM joint (never the
    #     gripper) into its mechanical hard stop and anchors the zero to the
    #     known stop angles (highest confidence, no jig needed). OPT-IN: the
    #     arm moves by itself during calibration — clear the workspace. The
    #     gripper zero must be flash-burned by hand at the closed pose first
    #     (openarm_gripper_zero.py). Requires config.side to be set.
    calibration_anchor: str = "hang_down"

    # Per-unit overrides for the bump-to-stop anchor (all optional; defaults in
    # lerobot.motors.damiao.damiao_bump_calibration are the official openarm_can
    # values and WILL need tuning per unit on the real arms):
    #   bump_stop_angles_deg: measured mechanical stop angles in the URDF frame
    #     (e.g. {"joint_1": -78.5}) when a unit's stop deviates from the nominal.
    #   bump_torque_thresholds_nm / bump_velocity_thresholds_deg_s: per-joint
    #     stop-contact detection thresholds (|tau| above / |vel| below).
    bump_stop_angles_deg: dict[str, float] | None = None
    bump_torque_thresholds_nm: dict[str, float] | None = None
    bump_velocity_thresholds_deg_s: dict[str, float] | None = None

    # --- Persistent zero (software homing offsets) ----------------------------
    # Whether connect() burns a new motor-side zero (Damiao 0xFE) at the current
    # physical pose.
    #   None (default, auto): legacy calibrations (homing offsets all zero) keep
    #     the historical per-session re-zero; calibrations captured with the new
    #     calibrate() flow carry non-zero homing offsets and skip the burn — the
    #     zero then persists across sessions/power cycles via the calibration
    #     file, so the arm does NOT have to be hung down identically at every
    #     startup.
    #   True: always re-zero at connect (forces the legacy behavior; clears any
    #     stored homing offsets for the session).
    #   False: never re-zero at connect (requires a valid homing-offset
    #     calibration or a motor zero known to be correct).
    rezero_on_connect: bool | None = None

    # When the per-session re-zero is skipped, connect() sanity-checks that all
    # joints read within joint_limits widened by this many degrees BEFORE
    # enabling the startup alignment move. Catches a stale frame (motor zero
    # reverted after power cycle, wrong calibration file, boot-time wrap) that
    # would otherwise drive the arm to a wrong physical pose. None disables.
    start_position_tolerance_deg: float | None = 30.0

    # --- Startup alignment (official AdjustPosition port) ---------------------
    # On connect, softly interpolate the arm from its current pose to the fixed
    # initial pose (all joints 0, elbow joint_4 = 36 deg, gripper 0) with gains
    # much softer than the teleop gains, mirroring the official openarm_teleop
    # startup. Leader and follower both aligning to the same pose means teleop
    # starts matched, with no manual pose alignment and no jump on the first
    # tracking command.
    align_on_connect: bool = True
    align_duration_s: float = 2.2

    # Startup alignment target. By default the arm aligns to the official
    # OPENARM_INITIAL_POSITION_DEG (all joints 0, elbow 36 deg). To start teleop
    # from a custom "ready" pose (e.g. elbow bent + arm raised), point
    # initial_pose_path at a per-side YAML captured with
    # scripts/capture_initial_pose.sh; initial_pose_deg overrides it inline.
    # Missing joints fall back to the default; values are clamped to joint_limits.
    initial_pose_path: str | None = None
    initial_pose_deg: dict[str, float] | None = None

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Motor configuration for OpenArms (7 DOF per arm)
    # Maps motor names to (send_can_id, recv_can_id, motor_type)
    # Based on: https://docs.openarm.dev/software/setup/configure-test
    # OpenArms uses 4 types of motors:
    # - DM8009 (DM-J8009P-2EC) for shoulders (high torque)
    # - DM4340P and DM4340 for shoulder rotation and elbow
    # - DM4310 (DM-J4310-2EC V1.1) for wrist and gripper
    motor_config: dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11, "dm8009"),  # J1 - Shoulder pan (DM8009)
            "joint_2": (0x02, 0x12, "dm8009"),  # J2 - Shoulder lift (DM8009)
            "joint_3": (0x03, 0x13, "dm4340"),  # J3 - Shoulder rotation (DM4340)
            "joint_4": (0x04, 0x14, "dm4340"),  # J4 - Elbow flex (DM4340)
            "joint_5": (0x05, 0x15, "dm4310"),  # J5 - Wrist roll (DM4310)
            "joint_6": (0x06, 0x16, "dm4310"),  # J6 - Wrist pitch (DM4310)
            "joint_7": (0x07, 0x17, "dm4310"),  # J7 - Wrist rotation (DM4310)
            "gripper": (0x08, 0x18, "dm4310"),  # J8 - Gripper (DM4310)
        }
    )

    # MIT control parameters for position control (used in send_action)
    # List of 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(
        default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 25.0]
    )
    position_kd: list[float] = field(default_factory=lambda: [3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])

    # Values for joint limits. Can be overridden via CLI (for custom values) or by setting config.side to either 'left' or 'right'.
    # If config.side is left set to None and no CLI values are passed, the default joint limit values are small for safety.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "joint_1": (-5.0, 5.0),
            "joint_2": (-5.0, 5.0),
            "joint_3": (-5.0, 5.0),
            "joint_4": (0.0, 5.0),
            "joint_5": (-5.0, 5.0),
            "joint_6": (-5.0, 5.0),
            "joint_7": (-5.0, 5.0),
            "gripper": (-5.0, 0.0),
        }
    )


@RobotConfig.register_subclass("openarm_follower")
@dataclass
class OpenArmFollowerConfig(RobotConfig, OpenArmFollowerConfigBase):
    pass
