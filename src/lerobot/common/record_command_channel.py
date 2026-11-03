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
"""Command channel for ``lerobot-record-cmd``: next / discard / return / stop and the recorder's state.

A page or a script sends one JSON request per command to a local ZMQ REP socket (default
``tcp://127.0.0.1:5730``; ``LEROBOT_RECORD_CMD`` changes it, ``off`` disables it) and gets one JSON reply::

    {"cmd": "next"}     -> the right arrow    (exit_early)
    {"cmd": "discard"}  -> the left arrow     (rerecord_episode + exit_early)
    {"cmd": "return"}   -> the 'a' key        (align_start_pose)
    {"cmd": "stop"}     -> esc                (stop_recording + exit_early): an ordinary end of collection
    {"cmd": "state"}    -> {"phase", "episode", "saved", "num_episodes", "events", ...}; changes nothing

The commands set exactly the flags the keys set, in the same ``events`` dict, and print the same lines,
so the record loop cannot tell a command from a key press and anything that reads the console keeps working.
The keyboard stays active next to the channel.

**Not an emergency stop.** ``stop`` ends the collection the normal way (the arms may return along the
start-pose waypoints before torque-off). The emergency stop is unchanged: Ctrl-C in the recorder's terminal
releases torque immediately and does not go through this channel; the hardware e-stop stays the first line.

Client side::

    python -m lerobot.common.record_command_channel state
    python -m lerobot.common.record_command_channel next --endpoint tcp://127.0.0.1:5730
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import Any

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5730"
ENDPOINT_ENV = "LEROBOT_RECORD_CMD"

# command -> (flags set to True, the line the matching key prints)
COMMANDS: dict[str, tuple[tuple[str, ...], str]] = {
    "next": (("exit_early",), "Right arrow key pressed. Exiting loop..."),
    "discard": (
        ("rerecord_episode", "exit_early"),
        "Left arrow key pressed. Exiting loop and rerecord the last episode...",
    ),
    "return": (("align_start_pose",), "'a' key pressed. Aligning the arm to the start pose..."),
    "stop": (("stop_recording", "exit_early"), "Escape key pressed. Stopping data recording..."),
}


def endpoint_from_env() -> str | None:
    """The endpoint to bind, or None when the channel is switched off."""
    value = os.environ.get(ENDPOINT_ENV, DEFAULT_ENDPOINT).strip()
    return None if value.lower() in ("", "off", "false", "0") else value


class RecorderState:
    """What the recorder is doing, as far as the wrapper can see it (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._d: dict[str, Any] = {
            "phase": "connecting",  # connecting -> waiting <-> recording -> done | error
            "episode": None,  # index of the take being recorded (or the next one while waiting)
            "num_episodes": None,
            "task": None,
            "started_at": time.time(),
            "last_command": None,
        }
        self._dataset = None

    def update(self, **kw: Any) -> None:
        with self._lock:
            self._d.update(kw)
            self._d["updated_at"] = time.time()

    def set_dataset(self, dataset: Any) -> None:
        with self._lock:
            self._dataset = dataset

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            d = dict(self._d)
            ds = self._dataset
        if ds is not None:
            try:
                # episodes saved so far in this dataset (incl. resumed ones)
                d["saved"] = int(ds.num_episodes)
            except Exception:  # noqa: BLE001 - a state query must never fail
                d["saved"] = None
        return d


class CommandChannel:
    """ZMQ REP socket served from a daemon thread. Bind early (before the robot connects); attach the
    record loop's ``events`` dict once the keyboard listener has created it."""

    def __init__(self, endpoint: str, state: RecorderState | None = None):
        self.endpoint = endpoint
        self.state = state or RecorderState()
        self.events: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock = None
        self._ctx = None

    # ---- lifecycle ----------------------------------------------------------------------------------------
    def bind(self) -> CommandChannel:
        """Bind now, so a busy port fails before anything moves; serve requests from a daemon thread."""
        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        try:
            self._sock.bind(self.endpoint)
        except zmq.ZMQError as e:
            self._sock.close()
            raise RuntimeError(
                f"record command channel: cannot bind {self.endpoint} ({e}). Another recorder running? "
                f"Set {ENDPOINT_ENV}=<endpoint> or {ENDPOINT_ENV}=off."
            ) from e
        self._thread = threading.Thread(target=self._serve, name="record-command-channel", daemon=True)
        self._thread.start()
        logging.info(
            "Record command channel on %s (next / discard / return / stop / state). "
            "Ctrl-C stays the immediate stop; the keyboard keys keep working.",
            self.endpoint,
        )
        return self

    def attach(self, events: dict) -> None:
        self.events = events
        self.state.update(phase="waiting")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    # ---- requests -----------------------------------------------------------------------------------------
    def handle(self, request: Any) -> dict[str, Any]:
        """One request -> one reply. Accepts {"cmd": "..."} or a bare command string."""
        cmd = request.get("cmd") if isinstance(request, dict) else request
        if not isinstance(cmd, str):
            return {"ok": False, "error": 'expected {"cmd": "next|discard|return|stop|state"}'}
        cmd = cmd.strip().lower()
        if cmd == "state":
            return {"ok": True, "state": self._state()}
        if cmd not in COMMANDS:
            return {"ok": False, "error": f"unknown command {cmd!r}", "commands": [*COMMANDS, "state"]}
        if self.events is None:
            return {"ok": False, "error": "recorder not ready (still connecting)", "state": self._state()}
        if self.state.snapshot().get("phase") in ("done", "error"):
            return {"ok": False, "error": "recording has ended", "state": self._state()}
        flags, line = COMMANDS[cmd]
        print(f"{line} (command channel)", flush=True)
        for flag in flags:
            self.events[flag] = True
        self.state.update(last_command={"cmd": cmd, "t": time.time()})
        return {"ok": True, "cmd": cmd, "state": self._state()}

    def _state(self) -> dict[str, Any]:
        s = self.state.snapshot()
        s["events"] = dict(self.events) if self.events is not None else None
        return s

    def _serve(self) -> None:
        import zmq

        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                if not dict(poller.poll(100)).get(self._sock):
                    continue
                raw = self._sock.recv()
                try:
                    req = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    req = raw.decode("utf-8", "replace")
                try:
                    reply = self.handle(req)
                except Exception as e:  # noqa: BLE001 - never kill the serving thread
                    reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self._sock.send_string(json.dumps(reply, default=str))
            except zmq.ZMQError:
                if self._stop.is_set():
                    break
                logging.exception("record command channel: socket error")
                time.sleep(0.1)


def send(cmd: str, endpoint: str = DEFAULT_ENDPOINT, timeout_s: float = 2.0) -> dict[str, Any]:
    """Client: send one command and return the reply (raises TimeoutError when nobody answers)."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    try:
        sock.connect(endpoint)
        sock.send_string(json.dumps({"cmd": cmd}))
        try:
            return json.loads(sock.recv_string())
        except zmq.Again as e:
            raise TimeoutError(
                f"no reply from {endpoint} within {timeout_s} s (recorder not running?)"
            ) from e
    finally:
        sock.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Send one command to a running lerobot-record-cmd.")
    ap.add_argument("cmd", choices=[*COMMANDS, "state"])
    ap.add_argument("--endpoint", default=os.environ.get(ENDPOINT_ENV, DEFAULT_ENDPOINT))
    ap.add_argument("--timeout-s", type=float, default=2.0)
    a = ap.parse_args(argv)
    try:
        reply = send(a.cmd, a.endpoint, a.timeout_s)
    except TimeoutError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 2
    print(json.dumps(reply, ensure_ascii=False, default=str))
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
