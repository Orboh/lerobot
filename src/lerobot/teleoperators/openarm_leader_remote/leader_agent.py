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

"""Operator-side half of the OpenArm remote teleop (design doc
"openarm-remote-teleop-design").

Runs on the PC physically attached to the LEADER arm. Reads the leader at
`fps` (gravity compensation is executed locally inside
`OpenArmLeader.get_action()`, so it never crosses the network) and streams the
action dict to the follower host over ZMQ PUSH (CONFLATE=1: only the newest
sample is ever delivered).

Start order does not matter: PUSH reconnects in the background, sends are
dropped (and counted) until the follower host is listening.

SAFETY: Ctrl-C disconnects the leader, which DISABLES ITS TORQUE — the leader
arm goes limp and falls under its own weight. Lower the leader to a safe
hanging pose before stopping. The follower side is unaffected (it holds).

Usage (real arm):
    python -m lerobot.teleoperators.openarm_leader_remote.leader_agent \
        --remote_host=<follower-host-ip> \
        --teleop.port=can_lead_r --teleop.id=leader_right \
        --teleop.gravity_compensation=true \
        --teleop.gravity_urdf_path=/path/to/openarm_v10_bimanual.urdf \
        --teleop.gravity_side=right

Transport-only smoke test without hardware (localhost only, sends a sine wave):
    python -m lerobot.teleoperators.openarm_leader_remote.leader_agent \
        --remote_host=127.0.0.1 --fake=true
"""

import json
import logging
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field

import draccus

from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from ..openarm_leader import OpenArmLeader, OpenArmLeaderConfig

logger = logging.getLogger(__name__)

FAKE_MOTORS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]


@dataclass
class LeaderAgentConfig:
    # Follower host (e.g. Thor) IP or hostname. The follower binds; we connect.
    remote_host: str

    # Physical leader configuration (same fields as --teleop.* of the local
    # teleop scripts; gravity compensation runs here, locally).
    teleop: OpenArmLeaderConfig = field(
        default_factory=lambda: OpenArmLeaderConfig(port="can_lead_r", id="leader_right")
    )

    remote_port: int = 5701
    echo_port: int = 5711
    arm: str = "right"
    fps: int = 200

    # Send a synthetic sine action instead of reading hardware. Transport
    # test only — refused for non-localhost targets so a fake stream can
    # never drive a real follower across the network.
    fake: bool = False

    # Stop after this many seconds (None = run until Ctrl-C). NOTE: shutdown
    # disables leader torque — the arm goes limp, same as Ctrl-C.
    duration_s: float | None = None

    log_every_s: float = 5.0


def _fake_action(t: float, use_velocity_and_torque: bool) -> dict[str, float]:
    action: dict[str, float] = {}
    for i, motor in enumerate(FAKE_MOTORS):
        action[f"{motor}.pos"] = 10.0 * math.sin(2.0 * math.pi * 0.2 * t + i)
        if use_velocity_and_torque:
            action[f"{motor}.vel"] = 0.0
            action[f"{motor}.torque"] = 0.0
    return action


@draccus.wrap()
def main(cfg: LeaderAgentConfig):
    import zmq  # lazy: optional dependency

    init_logging()

    if cfg.fake and cfg.remote_host not in ("127.0.0.1", "localhost"):
        raise ValueError(
            "--fake=true is a transport smoke test and only allowed against localhost; "
            f"got remote_host={cfg.remote_host!r}. A fake stream must never drive a real follower."
        )

    leader: OpenArmLeader | None = None
    if not cfg.fake:
        leader = OpenArmLeader(cfg.teleop)
        logger.info("Connecting leader (the arm will move to its initial pose)...")
        leader.connect()
        logger.warning(
            "Ctrl-C will disable leader torque (the arm goes limp). "
            "Lower the leader before stopping."
        )

    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.CONFLATE, 1)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(f"tcp://{cfg.remote_host}:{cfg.remote_port}")

    echo = ctx.socket(zmq.PULL)
    echo.setsockopt(zmq.CONFLATE, 1)
    echo.connect(f"tcp://{cfg.remote_host}:{cfg.echo_port}")

    logger.info(
        f"leader_agent up: -> tcp://{cfg.remote_host}:{cfg.remote_port} "
        f"(echo {cfg.echo_port}) arm={cfg.arm} fps={cfg.fps} fake={cfg.fake}"
    )

    seq = 0
    sent = 0
    dropped = 0
    rtts_ms: deque[float] = deque(maxlen=2000)
    window_t0 = time.perf_counter()
    window_loops = 0
    t_start = time.monotonic()

    try:
        while cfg.duration_s is None or time.monotonic() - t_start < cfg.duration_s:
            loop_start = time.perf_counter()
            now = time.monotonic()

            action = _fake_action(now - t_start, cfg.teleop.use_velocity_and_torque) if cfg.fake else leader.get_action()

            seq += 1
            msg = {"seq": seq, "t_mono": now, "arm": cfg.arm, "action": {k: (None if v is None else float(v)) for k, v in action.items()}}
            try:
                push.send_string(json.dumps(msg), flags=zmq.NOBLOCK)
                sent += 1
            except zmq.Again:
                dropped += 1

            # Drain RTT echoes (best-effort).
            while True:
                try:
                    e = json.loads(echo.recv_string(zmq.NOBLOCK))
                    if e.get("t_mono") is not None:
                        rtts_ms.append((time.monotonic() - float(e["t_mono"])) * 1e3)
                except zmq.Again:
                    break
                except (json.JSONDecodeError, TypeError, ValueError):
                    break

            window_loops += 1
            window_dt = time.perf_counter() - window_t0
            if window_dt >= cfg.log_every_s:
                rate = window_loops / window_dt
                if rtts_ms:
                    samples = sorted(rtts_ms)
                    p50 = statistics.median(samples)
                    p99 = samples[min(len(samples) - 1, int(0.99 * len(samples)))]
                    rtt_str = f"rtt p50={p50:.1f}ms p99={p99:.1f}ms (n={len(samples)})"
                else:
                    rtt_str = "rtt n/a (no echo yet)"
                logger.info(f"loop {rate:.0f}Hz sent={sent} dropped={dropped} {rtt_str}")
                window_t0 = time.perf_counter()
                window_loops = 0

            precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - loop_start), 0.0))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt: shutting down leader_agent.")
    finally:
        if leader is not None and leader.is_connected:
            logger.warning("Disabling leader torque — the arm goes limp now.")
            leader.disconnect()
        push.close()
        echo.close()
        ctx.term()


if __name__ == "__main__":
    main()
