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
"""2026-10-24: during recording the leader and the follower can be driven at the same time
(connect-time alignment, the per-episode start-pose return, and the return on disconnect).
Together with ``parallel_arms`` all four OpenArm arms move at once instead of one pair after
the other. Sequential stays the default, and a leader that must not be driven by software
keeps the old behaviour: nothing moves, so the follower never jumps to a pose the leader
is not at when teleoperation resumes."""

import threading

import pytest

import lerobot.common.control_utils as control_utils
from lerobot.scripts.lerobot_record import _connect_devices, _disconnect_devices

TARGET = {"joint_1.pos": 10.0, "joint_2.pos": 20.0}


class _Leader:
    def __init__(self, barrier=None, drivable=True, timeline=None):
        self.barrier = barrier
        self.drivable = drivable
        self.timeline = timeline if timeline is not None else []
        self.poses = []

    def can_soft_move(self):
        return self.drivable

    def soft_move_to(self, pose, duration_s=2.0):
        if not self.drivable:
            return False
        self.timeline.append("leader")
        if self.barrier is not None:
            self.barrier.wait()
        self.poses.append(pose)
        return True


class _Follower:
    def __init__(self, timeline=None):
        self.timeline = timeline if timeline is not None else []

    def get_observation(self):
        return {"joint_1.pos": 0.0, "joint_2.pos": 0.0, "image": None}


@pytest.fixture
def follower_moves(monkeypatch):
    moves = []
    state = {"barrier": None}

    def fake_move(robot, current, target, duration_s):
        robot.timeline.append("follower")
        if state["barrier"] is not None:
            state["barrier"].wait()
        moves.append((current, target))

    monkeypatch.setattr(control_utils, "follower_smooth_move_to", fake_move)
    yield moves, state
    control_utils.set_parallel_leader_follower(False)


def test_default_drives_leader_then_follower(follower_moves):
    moves, _ = follower_moves
    timeline = []
    control_utils.set_parallel_leader_follower(False)
    assert control_utils.drive_to_start_pose(_Follower(timeline), _Leader(timeline=timeline), TARGET)
    assert timeline == ["leader", "follower"]
    assert moves == [({"joint_1.pos": 0.0, "joint_2.pos": 0.0}, TARGET)]


def test_parallel_overlaps_leader_and_follower(follower_moves):
    moves, state = follower_moves
    # Each side waits for the other to have started: only possible if they run at once.
    barrier = threading.Barrier(2, timeout=5)
    state["barrier"] = barrier
    leader = _Leader(barrier=barrier)
    control_utils.set_parallel_leader_follower(True)
    assert control_utils.drive_to_start_pose(_Follower(), leader, TARGET)
    assert leader.poses == [{"joint_1": 10.0, "joint_2": 20.0}]
    assert moves == [({"joint_1.pos": 0.0, "joint_2.pos": 0.0}, TARGET)]


def test_parallel_never_moves_the_follower_alone_when_the_leader_refuses(follower_moves):
    moves, _ = follower_moves
    control_utils.set_parallel_leader_follower(True)
    assert not control_utils.drive_to_start_pose(_Follower(), _Leader(drivable=False), TARGET)
    assert moves == []


class _Device:
    def __init__(self, name, barrier=None, fail=False, connected=True):
        self.name = name
        self.barrier = barrier
        self.fail = fail
        self.is_connected = connected
        self.calls = []

    def _run(self, what):
        self.calls.append(what)
        if self.barrier is not None:
            self.barrier.wait()
        if self.fail:
            raise ConnectionError(f"{self.name} {what} failed")

    def connect(self):
        self._run("connect")

    def disconnect(self):
        self._run("disconnect")


def test_connect_and_disconnect_overlap_when_parallel():
    barrier = threading.Barrier(2, timeout=5)
    robot, teleop = _Device("robot", barrier), _Device("teleop", barrier)
    _connect_devices(robot, teleop, parallel=True)
    _disconnect_devices(robot, teleop, parallel=True)
    assert robot.calls == teleop.calls == ["connect", "disconnect"]


def test_sequential_connect_keeps_robot_first_and_stops_on_failure():
    robot, teleop = _Device("robot", fail=True), _Device("teleop")
    with pytest.raises(ConnectionError, match="robot connect failed"):
        _connect_devices(robot, teleop, parallel=False)
    assert teleop.calls == []


def test_parallel_disconnect_finishes_the_healthy_side_and_skips_unconnected():
    robot, teleop = _Device("robot", fail=True), _Device("teleop")
    with pytest.raises(ConnectionError, match="robot disconnect failed"):
        _disconnect_devices(robot, teleop, parallel=True)
    assert teleop.calls == ["disconnect"]

    idle = _Device("teleop", connected=False)
    _disconnect_devices(_Device("robot"), idle, parallel=True)
    _disconnect_devices(_Device("robot"), None, parallel=True)
    assert idle.calls == []
