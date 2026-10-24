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

import logging
from functools import cached_property

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.robot_utils import run_both_arms

from ..openarm_leader import OpenArmLeader, OpenArmLeaderConfig
from ..teleoperator import Teleoperator
from .config_bi_openarm_leader import BiOpenArmLeaderConfig

logger = logging.getLogger(__name__)


class BiOpenArmLeader(Teleoperator):
    """
    Bimanual OpenArm Leader Arms
    """

    config_class = BiOpenArmLeaderConfig
    name = "bi_openarm_leader"

    def __init__(self, config: BiOpenArmLeaderConfig):
        super().__init__(config)
        if self.id is None:
            raise ValueError(
                f"{self.name} requires an explicit id "
                "(e.g. --teleop.id=leader). Without one the calibration path "
                "collapses to 'None.json', which every unnamed OpenArm device "
                "would share regardless of side."
            )
        self.config = config

        left_arm_config = OpenArmLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            can_interface=config.left_arm_config.can_interface,
            use_can_fd=config.left_arm_config.use_can_fd,
            can_bitrate=config.left_arm_config.can_bitrate,
            can_data_bitrate=config.left_arm_config.can_data_bitrate,
            motor_config=config.left_arm_config.motor_config,
            manual_control=config.left_arm_config.manual_control,
            use_velocity_and_torque=config.left_arm_config.use_velocity_and_torque,
            position_kd=config.left_arm_config.position_kd,
            position_kp=config.left_arm_config.position_kp,
            gravity_compensation=config.left_arm_config.gravity_compensation,
            gravity_urdf_path=config.left_arm_config.gravity_urdf_path,
            gravity_side="left",
            gravity_scale=config.left_arm_config.gravity_scale,
            gravity_vector=config.left_arm_config.gravity_vector,
            align_on_connect=config.left_arm_config.align_on_connect,
            align_duration_s=config.left_arm_config.align_duration_s,
            return_on_disconnect=config.left_arm_config.return_on_disconnect,
            initial_pose_path=config.left_arm_config.initial_pose_path,
            initial_pose_deg=config.left_arm_config.initial_pose_deg,
            frozen_pose_path=config.left_arm_config.frozen_pose_path,
            frozen_gripper_deg=config.left_arm_config.frozen_gripper_deg,
            rezero_on_connect=config.left_arm_config.rezero_on_connect,
            start_position_tolerance_deg=config.left_arm_config.start_position_tolerance_deg,
            calibration_anchor=config.left_arm_config.calibration_anchor,
            bump_stop_angles_deg=config.left_arm_config.bump_stop_angles_deg,
            bump_torque_thresholds_nm=config.left_arm_config.bump_torque_thresholds_nm,
            bump_velocity_thresholds_deg_s=config.left_arm_config.bump_velocity_thresholds_deg_s,
            bilateral_feedback=config.left_arm_config.bilateral_feedback,
            bilateral_kp=config.left_arm_config.bilateral_kp,
            bilateral_kd=config.left_arm_config.bilateral_kd,
            bilateral_gripper_kp=config.left_arm_config.bilateral_gripper_kp,
            bilateral_gripper_kd=config.left_arm_config.bilateral_gripper_kd,
        )

        right_arm_config = OpenArmLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            can_interface=config.right_arm_config.can_interface,
            use_can_fd=config.right_arm_config.use_can_fd,
            can_bitrate=config.right_arm_config.can_bitrate,
            can_data_bitrate=config.right_arm_config.can_data_bitrate,
            motor_config=config.right_arm_config.motor_config,
            manual_control=config.right_arm_config.manual_control,
            use_velocity_and_torque=config.right_arm_config.use_velocity_and_torque,
            position_kd=config.right_arm_config.position_kd,
            position_kp=config.right_arm_config.position_kp,
            gravity_compensation=config.right_arm_config.gravity_compensation,
            gravity_urdf_path=config.right_arm_config.gravity_urdf_path,
            gravity_side="right",
            gravity_scale=config.right_arm_config.gravity_scale,
            gravity_vector=config.right_arm_config.gravity_vector,
            align_on_connect=config.right_arm_config.align_on_connect,
            align_duration_s=config.right_arm_config.align_duration_s,
            return_on_disconnect=config.right_arm_config.return_on_disconnect,
            initial_pose_path=config.right_arm_config.initial_pose_path,
            initial_pose_deg=config.right_arm_config.initial_pose_deg,
            frozen_pose_path=config.right_arm_config.frozen_pose_path,
            frozen_gripper_deg=config.right_arm_config.frozen_gripper_deg,
            rezero_on_connect=config.right_arm_config.rezero_on_connect,
            start_position_tolerance_deg=config.right_arm_config.start_position_tolerance_deg,
            calibration_anchor=config.right_arm_config.calibration_anchor,
            bump_stop_angles_deg=config.right_arm_config.bump_stop_angles_deg,
            bump_torque_thresholds_nm=config.right_arm_config.bump_torque_thresholds_nm,
            bump_velocity_thresholds_deg_s=config.right_arm_config.bump_velocity_thresholds_deg_s,
            bilateral_feedback=config.right_arm_config.bilateral_feedback,
            bilateral_kp=config.right_arm_config.bilateral_kp,
            bilateral_kd=config.right_arm_config.bilateral_kd,
            bilateral_gripper_kp=config.right_arm_config.bilateral_gripper_kp,
            bilateral_gripper_kd=config.right_arm_config.bilateral_gripper_kd,
        )

        self.left_arm = OpenArmLeader(left_arm_config)
        self.right_arm = OpenArmLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        run_both_arms(
            lambda: self.left_arm.connect(calibrate),
            lambda: self.right_arm.connect(calibrate),
            parallel=self.config.parallel_arms,
        )

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def can_soft_move(self) -> bool:
        """Whether ``soft_move_to`` would drive both leaders (see ``OpenArmLeader.can_soft_move``)."""
        return self.left_arm.can_soft_move() and self.right_arm.can_soft_move()

    def soft_move_to(self, pose_deg: dict[str, float], duration_s: float = 2.0) -> bool:
        """Softly drive both leaders to ``pose_deg`` (keys ``left_<motor>`` / ``right_<motor>``).

        Splits the pose by side prefix and calls each arm's ``OpenArmLeader.soft_move_to``
        in turn (left first, then right — the same order as ``connect``), or both at once with
        ``parallel_arms``. Exists because
        the record start-pose return (``drive_to_start_pose`` in lerobot.common.control_utils)
        looks for ``soft_move_to`` on the teleop and passes the robot's ``.pos`` keys with
        the suffix stripped, i.e. ``right_joint_1`` ... ``left_gripper``; without this method
        the bimanual leader silently stays put and only the operator's hands can return it.

        Keys without a side prefix are ambiguous for a bimanual leader and are ignored
        with a warning (``BiOpenArmFollower.send_action`` drops them the same way). A side
        with no keys is left untouched. Returns True only when every arm that received
        keys returned True; returns False without moving anything when no side received
        keys, or when an arm refuses (torque-off manual_control, see OpenArmLeader).
        Torque is left on afterwards for the same reason as the single-arm version.
        """
        left_pose = {k.removeprefix("left_"): v for k, v in pose_deg.items() if k.startswith("left_")}
        right_pose = {k.removeprefix("right_"): v for k, v in pose_deg.items() if k.startswith("right_")}
        unprefixed = [k for k in pose_deg if not (k.startswith("left_") or k.startswith("right_"))]
        if unprefixed:
            logger.warning(
                f"soft_move_to: ignoring keys without a left_/right_ prefix: {unprefixed}"
            )
        if not left_pose and not right_pose:
            logger.warning("soft_move_to: no left_/right_ keys in pose; nothing to drive.")
            return False
        left_ok, right_ok = run_both_arms(
            lambda: self.left_arm.soft_move_to(left_pose, duration_s) if left_pose else True,
            lambda: self.right_arm.soft_move_to(right_pose, duration_s) if right_pose else True,
            parallel=self.config.parallel_arms,
        )
        return bool(left_ok and right_ok)

    @check_if_not_connected
    def read_positions(self) -> dict[str, float | None]:
        """Both leaders' joint positions (``left_<motor>.pos`` / ``right_<motor>.pos``), read
        without changing either arm's torque command. See ``OpenArmLeader.read_positions``."""
        left = self.left_arm.read_positions()
        right = self.right_arm.read_positions()
        return {
            **{f"left_{k}": v for k, v in left.items()},
            **{f"right_{k}": v for k, v in right.items()},
        }

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict = {}

        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Split the follower's observation dict (``left_<motor>.pos`` /
        ``right_<motor>.pos``) by side and forward to each arm's
        ``OpenArmLeader.send_feedback`` (see its docstring and
        ``bilateral_feedback`` on the config — no-op unless enabled there).
        """
        left = {k.removeprefix("left_"): v for k, v in feedback.items() if k.startswith("left_")}
        right = {k.removeprefix("right_"): v for k, v in feedback.items() if k.startswith("right_")}
        if left:
            self.left_arm.send_feedback(left)
        if right:
            self.right_arm.send_feedback(right)

    @check_if_not_connected
    def disconnect(self) -> None:
        run_both_arms(self.left_arm.disconnect, self.right_arm.disconnect, parallel=self.config.parallel_arms)
