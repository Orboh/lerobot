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

from ..config import TeleoperatorConfig


@dataclass
class OpenArmLeaderConfigBase:
    """Base configuration for the OpenArms leader/teleoperator with Damiao motors."""

    # CAN interfaces - one per arm
    # Arm CAN interface (e.g., "can3")
    # Linux: "can0", "can1", etc.
    port: str

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
    can_interface: str = "socketcan"

    # CAN FD settings (OpenArms uses CAN FD by default)
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # Nominal bitrate (1 Mbps)
    can_data_bitrate: int = 5000000  # Data bitrate for CAN FD (5 Mbps)

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

    # Torque mode settings for manual control
    # When enabled, motors have torque disabled for manual movement
    manual_control: bool = True

    # --- Calibration anchor ----------------------------------------------------
    # Same semantics as OpenArmFollowerConfigBase.calibration_anchor:
    #   "hang_down" (default) captures the zero at the hand-positioned hang-down
    #   pose; "bump_to_stop" sweeps each ARM joint (never the gripper) into its
    #   mechanical hard stop and anchors the zero there (opt-in, the arm moves
    #   by itself). The leader has no `side` field: the bump sequence side is
    #   taken from `gravity_side` — set it explicitly when using bump_to_stop.
    calibration_anchor: str = "hang_down"

    # Per-unit bump-to-stop overrides (see OpenArmFollowerConfigBase for the
    # full note; defaults live in lerobot.motors.damiao.damiao_bump_calibration
    # and need per-unit tuning on the real arms).
    bump_stop_angles_deg: dict[str, float] | None = None
    bump_torque_thresholds_nm: dict[str, float] | None = None
    bump_velocity_thresholds_deg_s: dict[str, float] | None = None

    # --- Persistent zero (software homing offsets) ----------------------------
    # Same semantics as OpenArmFollowerConfigBase.rezero_on_connect:
    #   None (auto): legacy calibrations (offsets all zero) keep the historical
    #     per-session re-zero; homing-offset calibrations skip it (persistent
    #     zero, no need to hang the arm down at every startup).
    #   True: always re-zero at connect (legacy).  False: never.
    rezero_on_connect: bool | None = None

    # When the per-session re-zero is skipped AND the leader will apply torque
    # (gravity_compensation or MIT mode), connect() sanity-checks that all
    # joints read within the side's physical limits widened by this many
    # degrees before moving. None disables.
    start_position_tolerance_deg: float | None = 30.0

    # --- Startup alignment (official AdjustPosition port) ---------------------
    # On connect, softly interpolate the arm from its current pose to the fixed
    # initial pose (all joints 0, elbow joint_4 = 36 deg, gripper 0), mirroring
    # the official openarm_teleop startup so leader and follower start matched.
    # Needs torque, so it only runs in gravity_compensation or MIT/position
    # mode; pure manual_control (torque-off) skips it.
    align_on_connect: bool = True
    align_duration_s: float = 2.2

    # Startup alignment target (see OpenArmFollowerConfigBase for the full note).
    # Point initial_pose_path at a per-side YAML captured with
    # scripts/capture_initial_pose.sh to start from a custom "ready" pose;
    # initial_pose_deg overrides it inline. In gravity_compensation mode the
    # alignment injects gravity feed-forward so a raised pose is reached/held.
    initial_pose_path: str | None = None
    initial_pose_deg: dict[str, float] | None = None

    # When True, expose `.vel` and `.torque` per motor in action features.
    # Default False for compatibility with the position-only openarm_mini teleoperator.
    use_velocity_and_torque: bool = False

    # --- Gravity compensation -------------------------------------------------
    # Python/Pinocchio port of the C++ KDL reference
    # (openarm_teleop/control/gravity_compasation.cpp). When True, the leader runs
    # with torque ENABLED and injects, every cycle, the gravity feed-forward torque
    # G(q) so the arm is (near) weightless and can be moved by hand. This takes
    # precedence over `manual_control` for the torque branch in configure().
    gravity_compensation: bool = False

    # Path to the dynamics URDF (use the same file the C++ reference uses, e.g.
    # urdf/openarm_v10_bimanual.urdf). Required when gravity_compensation=True.
    gravity_urdf_path: str | None = None

    # Which arm this leader drives: "left" or "right". Selects the URDF chain
    # root=openarm_body_link0 -> leaf=openarm_<side>_hand and joints
    # openarm_<side>_joint1..7 (matches the C++ chain; gripper fingers excluded).
    gravity_side: str = "left"

    # Safety scale on the injected torque: tau_cmd = gravity_scale * G(q).
    # START LOW on real hardware (0.3) and ramp toward 1.0 only after confirming
    # the measured-q sign and zero match the model (see design note). 1.0 == full
    # compensation (the C++ uses no scale).
    gravity_scale: float = 0.3

    # Gravity vector in the chain-root (openarm_body_link0) frame. The C++ uses
    # (0, 0, -9.81); the root frame is Z-up (verified: world->body_link0 rotation
    # is identity in the URDF).
    gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # TODO(Steven, Pepijn): Not used ... ?
    # MIT control parameters (used when manual_control=False for torque control)
    # List of 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(
        default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0]
    )
    position_kd: list[float] = field(default_factory=lambda: [3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])


@TeleoperatorConfig.register_subclass("openarm_leader")
@dataclass
class OpenArmLeaderConfig(TeleoperatorConfig, OpenArmLeaderConfigBase):
    pass
