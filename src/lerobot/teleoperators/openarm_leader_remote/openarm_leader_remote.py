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

import json
import logging
import time
from typing import Any

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_openarm_leader_remote import OpenArmLeaderRemoteConfig

logger = logging.getLogger(__name__)


class OpenArmLeaderRemote(Teleoperator):
    """OpenArm leader whose joint stream arrives over the network.

    Drop-in Teleoperator for the follower host: `get_action()` returns the
    latest action dict published by the operator-side `leader_agent` (ZMQ
    PUSH/PULL, CONFLATE=1 so only the newest sample is ever queued — the
    LeKiwi transport pattern).

    Failure behavior: on packet staleness the last action is HELD (follower
    MIT control keeps position) and feed-forward keys are zeroed; on stream
    resume the output ramps to the live pose over `resume_ramp_s`.
    """

    config_class = OpenArmLeaderRemoteConfig
    name = "openarm_leader_remote"

    def __init__(self, config: OpenArmLeaderRemoteConfig):
        super().__init__(config)
        if self.id is None:
            raise ValueError(
                f"{self.name} requires an explicit id "
                "(e.g. --teleop.id=leader_right_remote). Without one the calibration path "
                "collapses to 'None.json', which every unnamed OpenArm device "
                "would share regardless of side."
            )
        self.config = config

        self._feature_keys = list(self.action_features.keys())
        self._zmq_context = None
        self._cmd_socket = None
        self._echo_socket = None

        # _target: latest merged action from the network.
        # _returned: what get_action() handed out last cycle (ramp start point).
        self._target: dict[str, float] = {}
        self._returned: dict[str, float] = {}
        self._last_recv_t: float = 0.0
        self._stalled: bool = False
        self._last_alarm_t: float = 0.0
        self._ramp_t0: float | None = None
        self._ramp_from: dict[str, float] = {}

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for motor in self.config.motor_names:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._cmd_socket is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Bind the ZMQ sockets and block until the first leader packet.

        The follower must never start "following" before a live leader stream
        exists; the first packet also seeds the held action so a stall in the
        very first cycles has something safe to hold.
        """
        import zmq  # lazy: optional dependency, only needed on the two teleop hosts

        self._zmq_context = zmq.Context()
        self._cmd_socket = self._zmq_context.socket(zmq.PULL)
        self._cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self._cmd_socket.bind(f"tcp://*:{self.config.zmq_port}")

        self._echo_socket = self._zmq_context.socket(zmq.PUSH)
        self._echo_socket.setsockopt(zmq.CONFLATE, 1)
        self._echo_socket.setsockopt(zmq.LINGER, 0)
        self._echo_socket.bind(f"tcp://*:{self.config.echo_port}")

        logger.info(
            f"{self} listening on tcp://*:{self.config.zmq_port} "
            f"(echo {self.config.echo_port}); waiting up to "
            f"{self.config.first_packet_timeout_s}s for the leader stream..."
        )
        poller = zmq.Poller()
        poller.register(self._cmd_socket, zmq.POLLIN)
        deadline = time.monotonic() + self.config.first_packet_timeout_s
        first = None
        while time.monotonic() < deadline:
            if poller.poll(timeout=200):
                first = self._recv_one()
                if first is not None:
                    break
        if first is None:
            self.disconnect()
            raise ConnectionError(
                f"{self.name}: no leader packet within {self.config.first_packet_timeout_s}s "
                f"on port {self.config.zmq_port}. Is leader_agent running and pointed at this host?"
            )

        self._apply_packet(first)
        self._returned = dict(self._target)
        missing = [k for k in self._feature_keys if k not in self._target]
        if missing:
            self.disconnect()
            raise ConnectionError(
                f"{self.name}: first packet is missing action keys {missing}; "
                "check the leader_agent side/arm configuration matches this teleoperator."
            )
        logger.info(f"{self} connected (first packet received, arm={first.get('arm')}).")

    @property
    def is_calibrated(self) -> bool:
        # Calibration lives with the physical leader on the operator PC.
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def _recv_one(self) -> dict[str, Any] | None:
        """Non-blocking receive of one packet; None when nothing is queued."""
        import zmq

        try:
            msg = self._cmd_socket.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            return None
        try:
            return dict(json.loads(msg))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"{self} dropped malformed packet: {e}")
            return None

    def _drain_latest(self) -> dict[str, Any] | None:
        """Return the newest queued packet (CONFLATE keeps at most one, the
        drain loop also covers the no-CONFLATE fallback of some transports)."""
        latest = None
        while True:
            pkt = self._recv_one()
            if pkt is None:
                return latest
            latest = pkt

    def _apply_packet(self, pkt: dict[str, Any]) -> None:
        action = pkt.get("action") or {}
        for key, value in action.items():
            if value is not None and key in self.action_features:
                self._target[key] = float(value)
        self._last_recv_t = time.monotonic()

    def _echo(self, pkt: dict[str, Any]) -> None:
        """Best-effort echo of seq/t_mono for operator-side RTT stats."""
        import zmq

        try:
            self._echo_socket.send_string(
                json.dumps({"seq": pkt.get("seq"), "t_mono": pkt.get("t_mono")}),
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _zero_feedforward(self) -> None:
        for key in list(self._target):
            if key.endswith((".vel", ".torque")):
                self._target[key] = 0.0

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.monotonic()
        pkt = self._drain_latest()

        if pkt is not None:
            gap = now - self._last_recv_t
            self._apply_packet(pkt)
            self._echo(pkt)
            if self._stalled:
                logger.warning(
                    f"{self} stream resumed after {gap:.2f}s; ramping to live pose "
                    f"over {self.config.resume_ramp_s}s."
                )
                self._stalled = False
                self._ramp_from = dict(self._returned)
                self._ramp_t0 = now
        else:
            staleness = now - self._last_recv_t
            if staleness > self.config.stale_warn_ms / 1000.0 and not self._stalled:
                self._stalled = True
                # Hold position, but stop pushing stale feed-forward.
                self._zero_feedforward()
                logger.warning(
                    f"{self} no packet for {staleness * 1e3:.0f}ms: holding last action "
                    "(torque stays ON; a limp arm would fall)."
                )
            if (
                self._stalled
                and staleness > self.config.stale_alarm_s
                and now - self._last_alarm_t > 1.0
            ):
                self._last_alarm_t = now
                logger.error(
                    f"{self} leader stream lost for {staleness:.1f}s — still holding. "
                    "Check Wi-Fi/AP and the leader_agent process."
                )

        out = dict(self._target)
        if self._ramp_t0 is not None:
            alpha = (now - self._ramp_t0) / self.config.resume_ramp_s
            if alpha >= 1.0:
                self._ramp_t0 = None
            else:
                for key, tgt in out.items():
                    held = self._ramp_from.get(key, tgt)
                    out[key] = held + (tgt - held) * alpha

        self._returned = dict(out)
        return out

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        raise NotImplementedError("Feedback is not implemented for OpenArm leader remote.")

    def disconnect(self) -> None:
        for sock_name in ("_cmd_socket", "_echo_socket"):
            sock = getattr(self, sock_name)
            if sock is not None:
                sock.close()
                setattr(self, sock_name, None)
        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None
        logger.info(f"{self} disconnected.")
