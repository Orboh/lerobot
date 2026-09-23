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
"""Start-pose resolution for OpenArm/Damiao. Pure logic -- no CAN hardware."""

import pytest

from lerobot.motors.damiao.damiao_alignment import (
    load_initial_pose_file,
    load_initial_pose_waypoints,
    resolve_initial_pose,
    resolve_initial_pose_sequence,
)

CAMERA_POSE = """
waypoints:
  - {joint_1: -69.38, joint_2: -1.5, joint_3: 0.26, joint_4: 74.73, joint_5: 1.04, joint_6: 1.11, joint_7: 0.0}
joint_1: 24.7
joint_2: 3.0
joint_3: 16.1
joint_4: 41.6
joint_5: 75.0
joint_6: 37.7
joint_7: -68.5
gripper: 0.0
"""

READY_POSE = """
joint_1: -69.38
joint_2: -1.5
joint_3: 0.26
joint_4: 74.73
joint_5: 1.04
joint_6: 1.11
joint_7: 0.0
gripper: 0.0
"""


@pytest.fixture
def camera_pose_file(tmp_path):
    p = tmp_path / "camera.yaml"
    p.write_text(CAMERA_POSE)
    return str(p)


@pytest.fixture
def ready_pose_file(tmp_path):
    p = tmp_path / "ready.yaml"
    p.write_text(READY_POSE)
    return str(p)


def test_flat_read_ignores_the_waypoints_list(camera_pose_file):
    """The flat joint mapping must still parse from a file that also has waypoints."""
    pose = load_initial_pose_file(camera_pose_file)
    assert pose["joint_1"] == 24.7
    assert "waypoints" not in pose


def test_no_waypoints_key_means_no_waypoints(ready_pose_file):
    assert load_initial_pose_waypoints(ready_pose_file) == []


def test_sequence_without_waypoints_matches_the_single_pose(ready_pose_file):
    """Regression guard: files captured before waypoints existed must behave exactly as before."""
    sequence = resolve_initial_pose_sequence(initial_pose_path=ready_pose_file)
    assert sequence == [resolve_initial_pose(initial_pose_path=ready_pose_file)]


def test_sequence_ends_at_the_final_pose(camera_pose_file):
    sequence = resolve_initial_pose_sequence(initial_pose_path=camera_pose_file)
    assert len(sequence) == 2
    assert sequence[-1] == resolve_initial_pose(initial_pose_path=camera_pose_file)
    assert sequence[0]["joint_1"] == -69.38
    assert sequence[0]["joint_4"] == 74.73


def test_waypoint_inherits_omitted_joints_from_the_final_pose(tmp_path):
    """A partial waypoint must move only the joints it names.

    Back-filling from the official default instead would yank the unnamed joints
    somewhere the operator never asked for -- the opposite of why a waypoint is
    there (staying in known-safe space on the way in).
    """
    p = tmp_path / "partial.yaml"
    p.write_text("waypoints:\n  - {joint_1: -69.38}\njoint_1: 24.7\njoint_5: 75.0\ngripper: 0.0\n")
    waypoint, final = resolve_initial_pose_sequence(initial_pose_path=str(p))
    assert waypoint["joint_1"] == -69.38
    assert waypoint["joint_5"] == final["joint_5"] == 75.0


def test_gripper_stays_pinned_to_zero_in_every_pose(camera_pose_file):
    """Commanding a non-zero gripper hold on startup is what damaged a follower gripper."""
    for pose in resolve_initial_pose_sequence(initial_pose_path=camera_pose_file):
        assert pose["gripper"] == 0.0


def test_joint_limits_clamp_waypoints_too(camera_pose_file):
    sequence = resolve_initial_pose_sequence(
        initial_pose_path=camera_pose_file, joint_limits={"joint_1": (-10.0, 10.0)}
    )
    assert sequence[0]["joint_1"] == -10.0
    assert sequence[-1]["joint_1"] == 10.0


def test_explicit_pose_override_skips_waypoints(camera_pose_file):
    """Programmatic callers pass a pose directly and must keep moving to it in one go."""
    sequence = resolve_initial_pose_sequence(
        initial_pose_deg={"joint_1": 5.0}, initial_pose_path=camera_pose_file
    )
    assert len(sequence) == 1
    assert sequence[0]["joint_1"] == 5.0


def test_malformed_waypoints_raise(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("waypoints: 3\njoint_1: 1.0\n")
    with pytest.raises(ValueError):
        load_initial_pose_waypoints(str(p))

    p2 = tmp_path / "bad2.yaml"
    p2.write_text("waypoints:\n  - 3\njoint_1: 1.0\n")
    with pytest.raises(ValueError):
        load_initial_pose_waypoints(str(p2))
