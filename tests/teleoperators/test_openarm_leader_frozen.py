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
"""Frozen-side action for the OpenArm leader. Pure logic -- no CAN hardware."""

import pytest

from lerobot.teleoperators.openarm_leader.config_openarm_leader import OpenArmLeaderConfigBase
from lerobot.teleoperators.openarm_leader.openarm_leader import OpenArmLeader

POSE = """
waypoints:
  - {joint_1: -69.38, joint_4: 74.73}
joint_1: 24.7
joint_2: 3.0
joint_3: 16.1
joint_4: 41.6
joint_5: 75.0
joint_6: 37.7
joint_7: -68.5
gripper: 0.0
"""

MOTORS = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]


class _Bus:
    motors = dict.fromkeys(MOTORS)


def _leader(**kw):
    """A leader instance without touching CAN: only config and bus are needed."""
    cfg = OpenArmLeaderConfigBase(port="can_lead_r", gravity_side="right", **kw)
    obj = object.__new__(OpenArmLeader)
    obj.config = cfg
    obj.bus = _Bus()
    return obj


@pytest.fixture
def pose_file(tmp_path):
    p = tmp_path / "rightcam.yaml"
    p.write_text(POSE)
    return str(p)


def test_live_side_is_not_frozen():
    assert _leader()._build_frozen_action() is None


def test_frozen_action_uses_the_final_pose_not_a_waypoint(pose_file):
    """The held pose is where the arm ends up, not somewhere on the way in."""
    action = _leader(frozen_pose_path=pose_file)._build_frozen_action()
    assert action["joint_1.pos"] == 24.7
    assert action["joint_4.pos"] == 41.6


def test_frozen_gripper_is_separate_from_the_pose_file(pose_file):
    """Alignment always pins the gripper closed, but a closed hand blocks the camera."""
    action = _leader(frozen_pose_path=pose_file, frozen_gripper_deg=-65.0)._build_frozen_action()
    assert action["gripper.pos"] == -65.0


def test_frozen_gripper_defaults_to_closed(pose_file):
    action = _leader(frozen_pose_path=pose_file)._build_frozen_action()
    assert action["gripper.pos"] == 0.0


def test_frozen_action_covers_every_motor(pose_file):
    action = _leader(frozen_pose_path=pose_file)._build_frozen_action()
    assert {k.removesuffix(".pos") for k in action if k.endswith(".pos")} == set(MOTORS)


def test_vel_and_torque_are_zero_when_exposed(pose_file):
    """A frozen arm must not inject leader velocity into the follower's MIT command."""
    action = _leader(frozen_pose_path=pose_file, use_velocity_and_torque=True)._build_frozen_action()
    assert all(action[f"{m}.vel"] == 0.0 and action[f"{m}.torque"] == 0.0 for m in MOTORS)


def test_vel_and_torque_absent_when_not_exposed(pose_file):
    action = _leader(frozen_pose_path=pose_file)._build_frozen_action()
    assert not any(k.endswith((".vel", ".torque")) for k in action)


def test_frozen_action_is_clamped_to_side_limits(tmp_path):
    p = tmp_path / "wild.yaml"
    p.write_text("joint_6: 999.0\ngripper: 0.0\n")
    action = _leader(frozen_pose_path=str(p))._build_frozen_action()
    assert action["joint_6.pos"] == 45.0  # RIGHT_DEFAULT_JOINTS_LIMITS joint_6 = (-45, 45)
