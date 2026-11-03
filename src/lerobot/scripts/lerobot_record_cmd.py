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
"""``lerobot-record`` plus a command channel (next / discard / return / stop / state).

Same arguments and the same behavior as ``lerobot-record``; the recording itself is the existing
``lerobot_record.record()``, called unchanged. This file only:

1. binds the command channel before anything connects (a busy port fails before the robot moves),
2. wraps ``init_keyboard_listener`` so the channel writes into the same ``events`` dict as the keys
   (the keys keep working as the fallback),
3. wraps ``record_loop`` to publish the recorder's phase (recording / waiting) and saved-episode count.

Ctrl-C is untouched: it reaches ``record()`` exactly as before and releases torque immediately; the
channel's ``stop`` is the ordinary end of collection (esc), never an emergency stop.

    lerobot-record-cmd <the usual lerobot-record arguments>
    LEROBOT_RECORD_CMD=tcp://127.0.0.1:5731 lerobot-record-cmd ...     # another endpoint
    python -m lerobot.common.record_command_channel next               # from a page or a script
"""

from __future__ import annotations

import functools
import logging

from lerobot.common.record_command_channel import CommandChannel, RecorderState, endpoint_from_env
from lerobot.scripts import lerobot_record as base
from lerobot.utils.import_utils import register_third_party_plugins


def install(channel: CommandChannel) -> None:
    """Point lerobot_record's keyboard listener and record loop at thin wrappers (module globals,
    looked up by ``record()`` at call time). Nothing else in lerobot_record changes."""
    state = channel.state
    init_keys = base.init_keyboard_listener
    loop = base.record_loop

    @functools.wraps(init_keys)
    def init_keyboard_listener_with_channel():
        listener, events = init_keys()
        channel.attach(events)
        return listener, events

    @functools.wraps(loop)
    def record_loop_with_state(*args, **kwargs):
        dataset = kwargs.get("dataset")
        if dataset is None:  # teleop slices between takes (start-pose cue, countdown): still "waiting"
            return loop(*args, **kwargs)
        state.set_dataset(dataset)
        state.update(phase="recording", episode=dataset.num_episodes, task=kwargs.get("single_task"))
        try:
            return loop(*args, **kwargs)
        finally:
            state.update(phase="waiting", episode=dataset.num_episodes)

    base.init_keyboard_listener = init_keyboard_listener_with_channel
    base.record_loop = record_loop_with_state


def main() -> None:
    register_third_party_plugins()
    endpoint = endpoint_from_env()
    if endpoint is None:
        logging.info("Record command channel off; plain lerobot-record.")
        base.record()
        return
    channel = CommandChannel(endpoint, RecorderState()).bind()
    install(channel)
    try:
        base.record()
        channel.state.update(phase="done")
    except BaseException:
        channel.state.update(phase="error")
        raise
    finally:
        channel.close()


if __name__ == "__main__":
    main()
