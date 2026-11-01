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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_openarm_leader_remote")
@dataclass
class BiOpenArmLeaderRemoteConfig(TeleoperatorConfig):
    """Network-fed bimanual OpenArm leader: two single-arm remote receivers in one teleoperator.

    Each arm has its own operator-side ``leader_agent`` (or PICO sender) and its own
    pair of ZMQ ports. The ports default to the ones ``scripts/run_leader_agent_left.sh``
    and ``run_leader_agent_right.sh`` already use (left 5702/5712, right 5701/5711), so
    both agents can run side by side.

    The settings are flat (not nested ``OpenArmLeaderRemoteConfig``) because that
    config is a registered choice class; nesting it would force a ``type`` selector
    on every nested field. The shared values below are passed to both arms as-is.
    Safety model is the single-arm one, applied per arm: on packet loss the last
    action is held (torque stays on), and on resume the output ramps back over
    ``resume_ramp_s``.
    """

    left_zmq_port: int = 5702
    left_echo_port: int = 5712
    right_zmq_port: int = 5701
    right_echo_port: int = 5711

    # connect() waits for the first packet of the left arm, then the right arm; each
    # wait is bounded by this. Start both senders before the record/teleoperate command.
    first_packet_timeout_s: float = 30.0
    stale_warn_ms: float = 250.0
    stale_alarm_s: float = 5.0
    resume_ramp_s: float = 1.0
    use_velocity_and_torque: bool = False

    motor_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "gripper",
        ]
    )
