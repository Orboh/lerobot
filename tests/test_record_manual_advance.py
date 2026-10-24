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
"""Manual episode advance: what the operator's keys do between episodes.

The record loop is driven end to end with the mock robot and teleop. The operator is
scripted off the state banners (each banner is the cue a real operator reacts to), and every
teleop / dataset call lands on one timeline so the order of events can be asserted:
whether the teleop loop ran while the arms were supposed to be held, when the take was
saved relative to the keys, and that a leader off the start pose blocks the start.
"""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import lerobot.common.control_utils as control_utils
import lerobot.scripts.lerobot_record as lerobot_record
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset
from lerobot.scripts.lerobot_record import RecordConfig, record
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

POSE = [10.0, 20.0, 30.0]


class Session:
    """One scripted recording session: the timeline plus the operator's reactions."""

    def __init__(self, monkeypatch, tmp_path, operator, **record_kwargs):
        self.timeline: list[tuple] = []
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "align_start_pose": False,
        }
        self.leader_offset = 0.0
        self.operator = operator
        self.tmp_path = tmp_path
        self.record_kwargs = record_kwargs

        def banner(title, keys):
            self.timeline.append(("banner", title))
            self.operator(self, title)

        monkeypatch.setattr(control_utils, "_banner", banner)
        monkeypatch.setattr(lerobot_record, "_banner", banner)
        monkeypatch.setattr(lerobot_record, "init_keyboard_listener", lambda: (None, self.events))
        # The follower walk is 1.5 s of real time; the order of calls is what matters here.
        monkeypatch.setattr(
            control_utils,
            "follower_smooth_move_to",
            lambda robot, current, target, **kw: self.timeline.append(("follower_move",)),
        )

        get_action = MockTeleop.get_action

        def traced_get_action(teleop):
            self.timeline.append(("get_action",))
            return get_action(teleop)

        def soft_move_to(teleop, pose_deg, duration_s=2.0):
            self.timeline.append(("leader_move",))
            self.leader_offset = 0.0
            return True

        def read_positions(teleop):
            return {f"{m}.pos": v + self.leader_offset for m, v in zip(teleop.motors, POSE, strict=True)}

        monkeypatch.setattr(MockTeleop, "get_action", traced_get_action)
        monkeypatch.setattr(MockTeleop, "soft_move_to", soft_move_to, raising=False)
        monkeypatch.setattr(MockTeleop, "read_positions", read_positions, raising=False)

        save_episode = LeRobotDataset.save_episode

        def traced_save(dataset, *args, **kwargs):
            self.timeline.append(("save", dataset.num_episodes))
            return save_episode(dataset, *args, **kwargs)

        monkeypatch.setattr(LeRobotDataset, "save_episode", traced_save)

    def press(self, key):
        self.timeline.append(("key", key))
        if key == "right":
            self.events["exit_early"] = True
        elif key == "left":
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
        elif key == "a":
            self.events["align_start_pose"] = True
        elif key == "esc":
            self.events["stop_recording"] = True
            self.events["exit_early"] = True

    def run(self, num_episodes=2):
        cfg = RecordConfig(
            robot=MockRobotConfig(random_values=False, static_values=POSE),
            teleop=MockTeleopConfig(random_values=False, static_values=POSE),
            dataset=DatasetRecordConfig(
                repo_id=DUMMY_REPO_ID,
                single_task="Dummy task",
                root=self.tmp_path / "record",
                num_episodes=num_episodes,
                episode_time_s=0.2,
                reset_time_s=0,
                push_to_hub=False,
            ),
            play_sounds=False,
            episode_advance="manual",
            start_pose_tolerance_deg=3.0,
            **self.record_kwargs,
        )
        return record(cfg)

    def index(self, entry, start=0):
        return self.timeline.index(entry, start)

    def banner_index(self, fragment, start=0):
        for i in range(start, len(self.timeline)):
            kind, *rest = self.timeline[i]
            if kind == "banner" and fragment in rest[0]:
                return i
        raise AssertionError(f"no banner containing {fragment!r} after {start}: {self.timeline}")

    def count(self, kind, lo, hi):
        return sum(1 for entry in self.timeline[lo:hi] if entry[0] == kind)


HELD = "両腕を開始姿勢で保持している"
WAITING = "待機中（録画していない）"
RETURNED = "腕は開始姿勢に戻っている"
BLOCKED = "leader が開始姿勢からずれている"
NOTHING_TO_DISCARD = "破棄できるテイクが無い"


def test_hold_keeps_the_teleop_loop_off_and_saves_on_return(monkeypatch, tmp_path):
    pressed_left = []

    def operator(session, title):
        if WAITING in title:
            session.press("a")
        elif NOTHING_TO_DISCARD in title:
            session.press("right")
        elif HELD in title and "保存済み" in title and not pressed_left:
            pressed_left.append(True)
            session.press("left")  # the take is already saved: must be ignored, not start a take
        elif HELD in title:
            session.press("right")

    s = Session(
        monkeypatch, tmp_path, operator, start_pose_hold=True, save_on_return=True, start_countdown_s=1
    )
    dataset = s.run(num_episodes=2)

    assert dataset.meta.total_episodes == 2

    # First episode: held straight from connect, so the teleop loop never ran before recording.
    first_rec = s.banner_index("録画中 episode 0")
    assert s.banner_index(HELD) < first_rec
    assert s.count("get_action", 0, first_rec) == 0

    # After episode 0 the loop runs (the operator can still move the arms) until 'a'.
    waiting = s.banner_index(WAITING, first_rec)
    key_a = s.index(("key", "a"), waiting)
    leader_move = s.index(("leader_move",), key_a)

    # 'a' returns the arms and the take is saved right there, before any right arrow.
    saved = s.index(("save", 0), leader_move)
    held_again = s.banner_index("保存済み", saved)
    second_rec = s.banner_index("録画中 episode 1", held_again)
    assert leader_move < saved < held_again < second_rec

    # From the return to the first frame of episode 1 nobody releases the leader, even though
    # a left arrow, a right arrow and the countdown all happen in between.
    assert s.count("get_action", leader_move, second_rec) == 0
    assert s.index(("key", "left"), held_again) < s.banner_index(NOTHING_TO_DISCARD, held_again) < second_rec

    # The last take is saved at the end as before; nothing was saved twice or discarded.
    assert [e for e in s.timeline if e[0] == "save"] == [("save", 0), ("save", 1)]


def test_default_flow_is_unchanged_apart_from_the_stray_left_arrow(monkeypatch, tmp_path):
    seen_waiting = []

    def operator(session, title):
        if WAITING in title and not seen_waiting:
            seen_waiting.append(True)
            session.press("left")  # nothing recorded yet: used to fall through into recording
        elif NOTHING_TO_DISCARD in title:
            session.press("right")
        elif WAITING in title and len(seen_waiting) == 1:
            seen_waiting.append(True)
            session.press("a")
        elif RETURNED in title:
            session.press("right")

    s = Session(monkeypatch, tmp_path, operator)
    dataset = s.run(num_episodes=2)

    assert dataset.meta.total_episodes == 2
    first_rec = s.banner_index("録画中 episode 0")
    assert s.banner_index(NOTHING_TO_DISCARD) < first_rec
    # Without the hold the teleop loop runs while waiting, before and after 'a'.
    assert s.count("get_action", 0, first_rec) > 0
    returned = s.banner_index(RETURNED, first_rec)
    second_rec = s.banner_index("録画中 episode 1", returned)
    # ... and the take is saved by the right arrow that starts the next episode.
    assert returned < s.index(("key", "right"), returned) < s.index(("save", 0)) < second_rec


def test_leader_moved_during_the_count_blocks_the_start(monkeypatch, tmp_path):
    def operator(session, title):
        if BLOCKED in title:
            if ("key", "right") in session.timeline[session.banner_index(BLOCKED) :]:
                session.press("a")
            else:
                session.press("right")  # refused while blocked
        elif HELD in title:
            session.press("right")

    s = Session(monkeypatch, tmp_path, operator, start_pose_hold=True, start_countdown_s=1)

    countdown = lerobot_record.countdown_to_episode_start
    counts = []

    def countdown_then_push(events, seconds, run_slice=None):
        ok = countdown(events, seconds, run_slice)
        if not counts:
            s.leader_offset = 40.0  # the operator's grip twists the held leader during the count
        counts.append(run_slice.__name__)
        return ok

    monkeypatch.setattr(lerobot_record, "countdown_to_episode_start", countdown_then_push)
    dataset = s.run(num_episodes=1)

    assert dataset.meta.total_episodes == 1
    blocked = s.banner_index(BLOCKED)
    rec = s.banner_index("録画中 episode 0")
    # No frame was recorded and the leader was never released until 'a' had returned it.
    assert blocked < s.index(("key", "a"), blocked) < s.index(("leader_move",), blocked) < rec
    assert s.count("get_action", 0, rec) == 0
    assert s.count("save", 0, rec) == 0
    # Held arms are only observed during either count, never driven by the teleop loop.
    assert counts == ["start_pose_hold_slice", "start_pose_hold_slice"]


def test_save_on_episode_end_saves_before_the_return_wait(monkeypatch, tmp_path):
    """2026-10-19: save_on_episode_end saves the instant record_loop ends (the right
    arrow/time limit that stopped THIS episode), before the 'a'-return wait even starts
    — earlier than save_on_return (which saves at the verified return) and the default
    (which saves at the right arrow that starts the NEXT episode). The take can then no
    longer be discarded with the left arrow (wait_for_episode_cue gets discardable=False)."""

    seen_waiting = []
    seen_discard_refused = []

    def operator(session, title):
        if WAITING in title and not seen_waiting:
            seen_waiting.append(True)
            session.press("left")  # pre-record: nothing to discard either way
        elif NOTHING_TO_DISCARD in title and not seen_discard_refused:
            seen_discard_refused.append(True)
            session.press("right")  # starts episode 0
        elif WAITING in title and len(seen_waiting) == 1:
            seen_waiting.append(True)
            session.press("left")  # post-record: take is already saved, must be refused
        elif NOTHING_TO_DISCARD in title:
            session.press("a")
        elif RETURNED in title:
            session.press("right")

    s = Session(monkeypatch, tmp_path, operator, save_on_episode_end=True)
    dataset = s.run(num_episodes=2)

    assert dataset.meta.total_episodes == 2
    first_rec = s.banner_index("録画中 episode 0")

    # Saved the instant episode 0's recording ends (here, the 0.2s time limit), before
    # the post-record wait shows its first banner.
    post_record_waiting = s.banner_index(WAITING, first_rec)
    saved = s.index(("save", 0))
    assert first_rec < saved < post_record_waiting

    # Left arrow during the post-record wait is refused: the take is already committed.
    discard_refused = s.banner_index(NOTHING_TO_DISCARD, post_record_waiting)
    assert post_record_waiting < s.index(("key", "left"), post_record_waiting) < discard_refused

    returned = s.banner_index(RETURNED, discard_refused)
    second_rec = s.banner_index("録画中 episode 1", returned)
    assert discard_refused < returned < second_rec

    # Nothing saved twice, nothing lost.
    assert [e for e in s.timeline if e[0] == "save"] == [("save", 0), ("save", 1)]


def press_after_frames(monkeypatch, session, n, key):
    """Press ``key`` once, after the ``n``-th frame of the first take (mid-recording)."""
    add_frame = LeRobotDataset.add_frame
    frames = []

    def traced_add_frame(dataset, *args, **kwargs):
        out = add_frame(dataset, *args, **kwargs)
        frames.append(True)
        if len(frames) == n:
            session.press(key)
        return out

    monkeypatch.setattr(LeRobotDataset, "add_frame", traced_add_frame)


def test_return_on_episode_end_saves_and_returns_without_a(monkeypatch, tmp_path):
    """2026-10-21: the right arrow that ends a take saves it and drives both arms back to
    the start pose by itself; the next right arrow only starts. 'a' is never needed."""

    def operator(session, title):
        if HELD in title:
            session.press("right")

    s = Session(monkeypatch, tmp_path, operator, start_pose_hold=True, return_on_episode_end=True)
    dataset = s.run(num_episodes=2)

    assert dataset.meta.total_episodes == 2
    first_rec = s.banner_index("録画中 episode 0")
    saved = s.index(("save", 0), first_rec)
    leader_move = s.index(("leader_move",), saved)
    held_again = s.banner_index(HELD, leader_move)
    second_rec = s.banner_index("録画中 episode 1", held_again)
    # Saved first, then returned, then held until the right arrow starts the next take.
    assert first_rec < saved < leader_move < held_again < second_rec
    assert s.count("get_action", leader_move, second_rec) == 0
    assert ("key", "a") not in s.timeline
    assert [e for e in s.timeline if e[0] == "save"] == [("save", 0), ("save", 1)]


def test_return_on_episode_end_left_arrow_discards_and_returns(monkeypatch, tmp_path):
    """The left arrow during a take discards it and returns the arms in one go; after the
    return the right arrow records the retake under the same episode index."""

    def operator(session, title):
        if HELD in title:
            session.press("right")

    s = Session(monkeypatch, tmp_path, operator, start_pose_hold=True, return_on_episode_end=True)
    press_after_frames(monkeypatch, s, 2, "left")
    dataset = s.run(num_episodes=1)

    assert dataset.meta.total_episodes == 1
    first_rec = s.banner_index("録画中 episode 0")
    left = s.index(("key", "left"), first_rec)
    discarded = s.banner_index("を破棄した", left)
    leader_move = s.index(("leader_move",), discarded)
    held_again = s.banner_index(HELD, leader_move)
    retake = s.banner_index("録画中 episode 0", held_again)
    assert first_rec < left < discarded < leader_move < held_again < retake
    # The discarded take was never written; only the retake was.
    assert s.count("save", 0, retake) == 0
    assert [e for e in s.timeline if e[0] == "save"] == [("save", 0)]
    assert ("key", "a") not in s.timeline


def test_esc_while_held_keeps_the_saved_take(monkeypatch, tmp_path):
    def operator(session, title):
        if WAITING in title:
            session.press("a")
        elif HELD in title and "保存済み" in title:
            session.press("esc")
        elif HELD in title:
            session.press("right")

    s = Session(monkeypatch, tmp_path, operator, start_pose_hold=True, save_on_return=True)
    dataset = s.run(num_episodes=3)

    assert dataset.meta.total_episodes == 1
    assert [e for e in s.timeline if e[0] == "save"] == [("save", 0)]
