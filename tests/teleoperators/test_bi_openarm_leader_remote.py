"""BiOpenArmLeaderRemote: two single-arm remote receivers behind one bimanual teleoperator.

Uses real ZMQ sockets on localhost (two fake senders, one per arm) so the port wiring,
key prefixing, per-arm hold-on-stale and soft_move_to splitting are exercised end to end.
"""

import json
import socket
import threading
import time

import pytest

zmq = pytest.importorskip("zmq")

from lerobot.teleoperators.bi_openarm_leader_remote import (  # noqa: E402
    BiOpenArmLeaderRemote,
    BiOpenArmLeaderRemoteConfig,
)

MOTORS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]


def _free_ports(n):
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


class FakeSender(threading.Thread):
    """Streams a constant action for one arm, like leader_agent / a PICO sender."""

    def __init__(self, port, arm, base):
        super().__init__(daemon=True)
        self.port, self.arm, self.base = port, arm, base
        self.stop_evt = threading.Event()
        self.paused = threading.Event()

    def run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://127.0.0.1:{self.port}")
        seq = 0
        while not self.stop_evt.is_set():
            if not self.paused.is_set():
                seq += 1
                action = {f"{m}.pos": self.base + i for i, m in enumerate(MOTORS)}
                sock.send_string(
                    json.dumps({"seq": seq, "t_mono": time.monotonic(), "arm": self.arm, "action": action})
                )
            time.sleep(0.005)
        sock.close()
        ctx.term()


@pytest.fixture
def rig():
    lz, le, rz, re_ = _free_ports(4)
    cfg = BiOpenArmLeaderRemoteConfig(
        id="t",
        left_zmq_port=lz,
        left_echo_port=le,
        right_zmq_port=rz,
        right_echo_port=re_,
        first_packet_timeout_s=5.0,
        stale_warn_ms=100.0,
    )
    left = FakeSender(lz, "left", 10.0)
    right = FakeSender(rz, "right", 100.0)
    left.start()
    right.start()
    teleop = BiOpenArmLeaderRemote(cfg)
    yield teleop, left, right
    teleop.disconnect()
    left.stop_evt.set()
    right.stop_evt.set()


def test_requires_id_and_distinct_ports():
    with pytest.raises(ValueError, match="explicit id"):
        BiOpenArmLeaderRemote(BiOpenArmLeaderRemoteConfig())
    with pytest.raises(ValueError, match="distinct"):
        BiOpenArmLeaderRemote(BiOpenArmLeaderRemoteConfig(id="t", left_zmq_port=5701, right_zmq_port=5701))


def test_action_features_are_prefixed_like_bi_openarm_leader():
    t = BiOpenArmLeaderRemote(BiOpenArmLeaderRemoteConfig(id="t"))
    keys = set(t.action_features)
    assert keys == {f"{s}_{m}.pos" for s in ("left", "right") for m in MOTORS}


def test_default_ports_match_leader_agent_scripts():
    c = BiOpenArmLeaderRemoteConfig(id="t")
    assert (c.left_zmq_port, c.left_echo_port) == (5702, 5712)
    assert (c.right_zmq_port, c.right_echo_port) == (5701, 5711)


def test_get_action_merges_both_arms(rig):
    teleop, _, _ = rig
    teleop.connect()
    assert teleop.is_connected
    time.sleep(0.1)
    a = teleop.get_action()
    assert a["left_joint_1.pos"] == pytest.approx(10.0)
    assert a["left_gripper.pos"] == pytest.approx(17.0)
    assert a["right_joint_1.pos"] == pytest.approx(100.0)
    assert a["right_gripper.pos"] == pytest.approx(107.0)


def test_one_arm_stalls_other_keeps_tracking_and_stalled_arm_holds(rig):
    teleop, left, right = rig
    teleop.connect()
    time.sleep(0.1)
    right.paused.set()
    time.sleep(0.4)  # beyond stale_warn_ms
    left.base = 20.0  # live arm moves on
    time.sleep(0.1)
    a = teleop.get_action()
    assert a["right_joint_1.pos"] == pytest.approx(100.0)  # held, not zeroed, not dropped
    assert a["left_joint_1.pos"] == pytest.approx(20.0)  # still live


def test_connect_failure_on_right_releases_left_port(rig):
    teleop, left, right = rig
    right.stop_evt.set()  # right never sends
    time.sleep(0.1)
    teleop.config.first_packet_timeout_s = 0.5
    teleop.right_arm.config.first_packet_timeout_s = 0.5
    with pytest.raises(ConnectionError):
        teleop.connect()
    assert not teleop.left_arm.is_connected
    assert not teleop.right_arm.is_connected


def test_soft_move_to_splits_by_side(monkeypatch):
    t = BiOpenArmLeaderRemote(BiOpenArmLeaderRemoteConfig(id="t"))
    calls = {}
    monkeypatch.setattr(t.left_arm, "soft_move_to", lambda p, d: calls.setdefault("left", p) is not None)
    monkeypatch.setattr(t.right_arm, "soft_move_to", lambda p, d: calls.setdefault("right", p) is not None)
    assert t.soft_move_to({"left_joint_1": 1.0, "right_joint_2": 2.0, "joint_3": 3.0})
    assert calls == {"left": {"joint_1": 1.0}, "right": {"joint_2": 2.0}}
    assert t.soft_move_to({"joint_3": 3.0}) is False  # no prefixed keys: nothing driven
