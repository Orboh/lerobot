#!/usr/bin/env python

# Copyright 2026 Orboh, Inc. All rights reserved.
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

from ..openarm_leader_remote import OpenArmLeaderRemote, OpenArmLeaderRemoteConfig
from ..teleoperator import Teleoperator
from .config_bi_openarm_leader_remote import BiOpenArmLeaderRemoteConfig

logger = logging.getLogger(__name__)


class BiOpenArmLeaderRemote(Teleoperator):
    """Bimanual OpenArm leader whose two joint streams arrive over the network.

    Composes two single-arm ``OpenArmLeaderRemote`` receivers the same way
    ``BiOpenArmLeader`` composes two local leaders, so ``action_features`` carry the
    same ``left_<motor>.pos`` / ``right_<motor>.pos`` keys and the bimanual follower
    accepts the action unchanged. All per-arm behavior (hold on stale packets,
    resume ramp, never torque-off) is the single-arm one.
    """

    config_class = BiOpenArmLeaderRemoteConfig
    name = "bi_openarm_leader_remote"

    def __init__(self, config: BiOpenArmLeaderRemoteConfig):
        super().__init__(config)
        if self.id is None:
            raise ValueError(
                f"{self.name} requires an explicit id "
                "(e.g. --teleop.id=leader_remote). Without one the per-arm ids collapse "
                "to 'None_left' / 'None_right'."
            )
        self.config = config

        shared = {
            "first_packet_timeout_s": config.first_packet_timeout_s,
            "stale_warn_ms": config.stale_warn_ms,
            "stale_alarm_s": config.stale_alarm_s,
            "resume_ramp_s": config.resume_ramp_s,
            "use_velocity_and_torque": config.use_velocity_and_torque,
            "motor_names": list(config.motor_names),
        }
        self.left_arm = OpenArmLeaderRemote(
            OpenArmLeaderRemoteConfig(
                id=f"{config.id}_left",
                calibration_dir=config.calibration_dir,
                zmq_port=config.left_zmq_port,
                echo_port=config.left_echo_port,
                **shared,
            )
        )
        self.right_arm = OpenArmLeaderRemote(
            OpenArmLeaderRemoteConfig(
                id=f"{config.id}_right",
                calibration_dir=config.calibration_dir,
                zmq_port=config.right_zmq_port,
                echo_port=config.right_echo_port,
                **shared,
            )
        )
        ports = {config.left_zmq_port, config.left_echo_port, config.right_zmq_port, config.right_echo_port}
        if len(ports) != 4:
            raise ValueError(
                "left/right zmq and echo ports must be four distinct ports, got "
                f"left {config.left_zmq_port}/{config.left_echo_port}, "
                f"right {config.right_zmq_port}/{config.right_echo_port}."
            )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Wait for the left stream, then the right one; undo the left bind if the right fails."""
        self.left_arm.connect(calibrate)
        try:
            self.right_arm.connect(calibrate)
        except Exception:
            self.left_arm.disconnect()
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        left = self.left_arm.get_action()
        right = self.right_arm.get_action()
        return {
            **{f"left_{k}": v for k, v in left.items()},
            **{f"right_{k}": v for k, v in right.items()},
        }

    def soft_move_to(self, pose_deg: dict[str, float], duration_s: float = 2.0) -> bool:
        """Ask each operator-side agent to drive its real leader to ``pose_deg``.

        Same contract as ``BiOpenArmLeader.soft_move_to``: keys are ``left_<motor>`` /
        ``right_<motor>`` without the ``.pos`` suffix; keys without a side prefix are
        ignored with a warning; a side with no keys is left alone. Left first, then
        right. Returns True only when every arm that received keys converged.
        """
        left_pose = {k.removeprefix("left_"): v for k, v in pose_deg.items() if k.startswith("left_")}
        right_pose = {k.removeprefix("right_"): v for k, v in pose_deg.items() if k.startswith("right_")}
        unprefixed = [k for k in pose_deg if not (k.startswith("left_") or k.startswith("right_"))]
        if unprefixed:
            logger.warning(f"soft_move_to: ignoring keys without a left_/right_ prefix: {unprefixed}")
        if not left_pose and not right_pose:
            logger.warning("soft_move_to: no left_/right_ keys in pose; nothing to drive.")
            return False
        ok = True
        if left_pose:
            ok = self.left_arm.soft_move_to(left_pose, duration_s) and ok
        if right_pose:
            ok = self.right_arm.soft_move_to(right_pose, duration_s) and ok
        return ok

    def send_feedback(self, feedback: dict) -> None:
        raise NotImplementedError("Feedback is not implemented for bimanual OpenArm leader remote.")

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
