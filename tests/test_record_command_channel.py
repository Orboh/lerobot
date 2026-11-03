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
"""lerobot-record-cmd: the command channel sets the same flags as the keys, the record loop is unchanged,
and Ctrl-C is not touched."""

import signal
import socket

import pytest

pytest.importorskip("zmq", reason="pyzmq is required (install lerobot[pyzmq-dep])")

from lerobot.common.record_command_channel import (  # noqa: E402
    COMMANDS,
    CommandChannel,
    RecorderState,
    endpoint_from_env,
    send,
)


def _events():
    return {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "align_start_pose": False,
    }


def _endpoint():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{s.getsockname()[1]}"


# The flags each key sets in lerobot.common.control_utils (pynput and stdin listeners).
KEY_FLAGS = {
    "next": {"exit_early"},
    "discard": {"rerecord_episode", "exit_early"},
    "return": {"align_start_pose"},
    "stop": {"stop_recording", "exit_early"},
}


@pytest.mark.parametrize("cmd", sorted(KEY_FLAGS))
def test_each_command_sets_exactly_the_flags_of_its_key(cmd, capsys):
    ch = CommandChannel("inproc://unused")
    ev = _events()
    ch.attach(ev)
    reply = ch.handle({"cmd": cmd})
    assert reply["ok"] and {k for k, v in ev.items() if v} == KEY_FLAGS[cmd]
    # the same console line as the key, so tools that read the recorder's output still work
    assert COMMANDS[cmd][1] in capsys.readouterr().out


def test_not_ready_unknown_ended_and_state():
    ch = CommandChannel("inproc://unused")
    assert not ch.handle("next")["ok"] and "not ready" in ch.handle("next")["error"]
    assert ch.handle("state")["state"]["phase"] == "connecting"
    ev = _events()
    ch.attach(ev)
    assert not ch.handle({"cmd": "torque_off"})["ok"]
    assert not ch.handle({"nope": 1})["ok"]
    st = ch.handle({"cmd": "state"})
    assert st["ok"] and st["state"]["phase"] == "waiting" and st["state"]["events"] == _events()
    assert not any(ev.values())  # a state query changes nothing
    ch.state.update(phase="done")
    assert not ch.handle("stop")["ok"] and not ev["stop_recording"]


def test_socket_roundtrip_busy_port_and_ctrl_c_untouched(capsys):
    before = signal.getsignal(signal.SIGINT)
    ep = _endpoint()
    ch = CommandChannel(ep, RecorderState()).bind()
    try:
        ev = _events()
        ch.attach(ev)
        assert send("return", ep)["ok"] and ev["align_start_pose"]
        assert send("state", ep)["state"]["events"]["align_start_pose"] is True
        with pytest.raises(RuntimeError, match="cannot bind"):
            CommandChannel(ep).bind()
        assert signal.getsignal(signal.SIGINT) is before  # the emergency stop path is not touched
    finally:
        ch.close()
    with pytest.raises(TimeoutError):
        send("state", ep, timeout_s=0.3)


def test_endpoint_env(monkeypatch):
    monkeypatch.delenv("LEROBOT_RECORD_CMD", raising=False)
    assert endpoint_from_env() == "tcp://127.0.0.1:5730"
    monkeypatch.setenv("LEROBOT_RECORD_CMD", "off")
    assert endpoint_from_env() is None
    monkeypatch.setenv("LEROBOT_RECORD_CMD", "tcp://127.0.0.1:5999")
    assert endpoint_from_env() == "tcp://127.0.0.1:5999"


def test_record_driven_only_through_the_channel(monkeypatch, tmp_path):
    """The real record() with the mock robot/teleop, the operator pressing nothing: every step is a command."""
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    import lerobot.scripts.lerobot_record as base
    from lerobot.scripts.lerobot_record_cmd import install
    from tests.test_record_manual_advance import RETURNED, WAITING, Session

    ep = _endpoint()
    sent, states = [], []

    def operator(session, title):
        def cmd(c):
            sent.append(c)
            session.timeline.append(("key", c))
            assert send(c, ep)["ok"]

        if WAITING in title:
            states.append(send("state", ep)["state"])
            cmd("return")
        elif RETURNED in title:
            cmd("next")

    s = Session(monkeypatch, tmp_path, operator)  # replaces the keyboard listener with its own events dict
    monkeypatch.setattr(base, "record_loop", base.record_loop)  # restored after the test
    ch = CommandChannel(ep, RecorderState()).bind()
    try:
        install(ch)
        dataset = s.run(num_episodes=2)
    finally:
        ch.close()

    assert dataset.meta.total_episodes == 2
    assert sent == ["return", "next", "return", "next"]
    assert [e for e in s.timeline if e[0] == "save"] == [("save", 0), ("save", 1)]
    # after the first take the recorder reports waiting, with nothing saved yet (the right arrow saves)
    after_first = states[1]
    assert after_first["phase"] == "waiting" and after_first["episode"] == 0 and after_first["saved"] == 0
    assert ch.state.snapshot()["saved"] == 2
