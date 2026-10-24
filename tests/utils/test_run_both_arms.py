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
"""2026-10-21: bimanual OpenArm connect / return / disconnect can drive both arms at once
(each arm is on its own CAN bus). Sequential stays the default; parallel must really
overlap the two sides, keep both results, and never swallow a failure of either side."""

import threading

import pytest

from lerobot.teleoperators.bi_openarm_leader.bi_openarm_leader import BiOpenArmLeader
from lerobot.utils.robot_utils import run_both_arms


def test_sequential_runs_left_then_right():
    order = []
    out = run_both_arms(lambda: order.append("left") or 1, lambda: order.append("right") or 2, parallel=False)
    assert order == ["left", "right"]
    assert out == (1, 2)


def test_parallel_overlaps_both_sides():
    # Each side waits for the other to have started: only possible if they run at once.
    barrier = threading.Barrier(2, timeout=5)
    out = run_both_arms(lambda: barrier.wait() >= 0, lambda: barrier.wait() >= 0, parallel=True)
    assert out == (True, True)


def test_parallel_failure_of_one_side_is_raised_after_both_finish():
    finished = []

    def left():
        raise ConnectionError("left handshake failed")

    def right():
        finished.append("right")
        return True

    with pytest.raises(ConnectionError, match="left handshake failed"):
        run_both_arms(left, right, parallel=True)
    # The healthy side was not abandoned mid-motion.
    assert finished == ["right"]


class _Arm:
    def __init__(self, barrier, ok=True):
        self.barrier = barrier
        self.ok = ok
        self.poses = []

    def soft_move_to(self, pose, duration_s):
        self.poses.append(pose)
        if self.barrier is not None:
            self.barrier.wait()
        return self.ok


def _leader(parallel, left, right):
    leader = object.__new__(BiOpenArmLeader)
    leader.config = type("Cfg", (), {"parallel_arms": parallel})()
    leader.left_arm = left
    leader.right_arm = right
    return leader


def test_leader_soft_move_to_drives_both_leaders_at_once():
    barrier = threading.Barrier(2, timeout=5)
    left, right = _Arm(barrier), _Arm(barrier)
    leader = _leader(True, left, right)
    assert leader.soft_move_to({"left_joint_1": 1.0, "right_joint_1": 2.0}) is True
    assert left.poses == [{"joint_1": 1.0}] and right.poses == [{"joint_1": 2.0}]


def test_leader_soft_move_to_reports_a_refusing_side_in_parallel():
    left, right = _Arm(None, ok=False), _Arm(None)
    leader = _leader(True, left, right)
    assert leader.soft_move_to({"left_joint_1": 1.0, "right_joint_1": 2.0}) is False
    assert right.poses == [{"joint_1": 2.0}]

