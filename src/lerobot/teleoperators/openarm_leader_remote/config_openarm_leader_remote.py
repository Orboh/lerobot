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


@TeleoperatorConfig.register_subclass("openarm_leader_remote")
@dataclass
class OpenArmLeaderRemoteConfig(TeleoperatorConfig):
    """Network-fed replacement for a locally attached OpenArm leader.

    Runs on the follower host (e.g. Thor). A `leader_agent` process on the
    operator PC reads the physical leader (gravity compensation stays local to
    that PC) and streams the action dict over ZMQ; this teleoperator consumes
    the stream and plugs into `lerobot-teleoperate` / `lerobot-record`
    unchanged via `--teleop.type=openarm_leader_remote`.

    Safety model (see the design doc "openarm-remote-teleop-design"):
    on packet loss the last action is held (the follower's MIT position hold),
    never torque-off; on resume the output ramps from the held pose to the
    live leader pose over `resume_ramp_s` to avoid a jump.
    """

    # ZMQ ports. The follower host BINDS both; the leader_agent CONNECTs to
    # them. cmd: leader_agent -> here (actions). echo: here -> leader_agent
    # (seq/t_mono echo for RTT measurement, best-effort).
    zmq_port: int = 5701
    echo_port: int = 5711

    # connect() blocks until the first action packet arrives (the follower
    # must not start "following" a leader that is not there yet).
    first_packet_timeout_s: float = 30.0

    # Staleness thresholds. Past stale_warn_ms the output freezes at the last
    # action (hold) and any .vel/.torque feed-forward keys are zeroed. Past
    # stale_alarm_s an error is logged repeatedly. Torque is NEVER disabled
    # here: a limp arm falls under its own weight.
    stale_warn_ms: float = 250.0
    stale_alarm_s: float = 5.0

    # After a stall, interpolate from the held action to the live leader pose
    # over this duration before resuming passthrough (jump prevention).
    resume_ramp_s: float = 1.0

    # Mirror of OpenArmLeaderConfig.use_velocity_and_torque: declares the
    # .vel/.torque keys in action_features. Phase 1 of the remote design runs
    # position-only (False); enable only after jitter has been measured.
    use_velocity_and_torque: bool = False

    # Expected motor names (defines action_features and filters inbound keys).
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
