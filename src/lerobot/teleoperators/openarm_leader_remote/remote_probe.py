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

"""Follower-side probe for the OpenArm remote teleop transport.

Consumes the leader_agent stream with the REAL `OpenArmLeaderRemote`
teleoperator (the exact class `lerobot-teleoperate` will use on the follower
host) but prints stream statistics instead of driving a robot. Use it to
verify transport, staleness/hold and resume-ramp behavior without any
hardware:

    # terminal 1 (fake sine stream):
    python -m lerobot.teleoperators.openarm_leader_remote.leader_agent \
        --remote_host=127.0.0.1 --fake=true
    # terminal 2:
    python -m lerobot.teleoperators.openarm_leader_remote.remote_probe --duration_s=15
"""

import logging
import time
from dataclasses import dataclass, field

import draccus

from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .config_openarm_leader_remote import OpenArmLeaderRemoteConfig
from .openarm_leader_remote import OpenArmLeaderRemote

logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    teleop: OpenArmLeaderRemoteConfig = field(
        default_factory=lambda: OpenArmLeaderRemoteConfig(id="leader_remote_probe")
    )
    fps: int = 200
    duration_s: float = 15.0
    print_every_s: float = 1.0


@draccus.wrap()
def main(cfg: ProbeConfig):
    init_logging()
    teleop = OpenArmLeaderRemote(cfg.teleop)
    teleop.connect()

    t0 = time.perf_counter()
    last_print = t0
    loops = 0
    window_loops = 0
    try:
        while time.perf_counter() - t0 < cfg.duration_s:
            loop_start = time.perf_counter()
            action = teleop.get_action()
            loops += 1
            window_loops += 1
            now = time.perf_counter()
            if now - last_print >= cfg.print_every_s:
                staleness_ms = (time.monotonic() - teleop._last_recv_t) * 1e3
                logger.info(
                    f"loop {window_loops / (now - last_print):.0f}Hz "
                    f"j1={action.get('joint_1.pos'):+8.3f} "
                    f"j4={action.get('joint_4.pos'):+8.3f} "
                    f"grip={action.get('gripper.pos'):+8.3f} "
                    f"staleness={staleness_ms:.0f}ms stalled={teleop._stalled}"
                )
                last_print = now
                window_loops = 0
            precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - loop_start), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
    logger.info(f"probe done: {loops} loops in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
