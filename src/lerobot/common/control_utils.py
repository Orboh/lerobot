# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

########################################################################################
# Utilities
########################################################################################
import logging
import math
import time
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.utils.import_utils import _deepdiff_available, require_package

if TYPE_CHECKING or _deepdiff_available:
    from deepdiff import DeepDiff
else:
    DeepDiff = None

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import Robot
from lerobot.types import PolicyAction


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception as e:
        # No GUI (plain SSH / DISPLAY unset): this is the normal case on Thor, not an error.
        # Control keys are read from the terminal's stdin instead (see _init_stdin_key_listener).
        logging.info(
            "GUI なし（headless）: 収集のキー（→ ← esc a）は端末の stdin から読む。"
            f"映像はモニターの rerun へ流す。（pynput import failed: {type(e).__name__}）"
        )
        logging.debug("pynput import traceback", exc_info=True)
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def _decode_control_key(data: bytes):
    """Map a raw key / escape sequence read from a TTY to a control-flow event name.

    Returns ``"next"`` (right arrow), ``"rerecord"`` (left arrow), ``"stop"`` (esc),
    or ``None``. Handles both CSI (``\\x1b[C``) and application-cursor (``\\x1bOC``)
    arrow encodings as well as a bare escape byte.
    """
    if not data:
        return None
    if data == b"\x1b":
        return "stop"  # bare escape
    if data[:1] == b"\x1b":
        last = data[-1:]
        if last == b"C":
            return "next"  # right arrow
        if last == b"D":
            return "rerecord"  # left arrow
    return None


def _init_stdin_key_listener(events):
    """Headless fallback that reads recording control keys from the controlling TTY.

    On Linux, pynput's listener needs a display (X/Wayland) and silently receives no
    events over a plain SSH session even when ``import pynput`` succeeds. When no
    display is available we instead put the terminal into cbreak mode and read the
    arrow keys / esc directly from stdin in a background thread, setting the same
    ``events`` flags as the pynput path (right arrow = next episode, left arrow =
    rerecord last episode, esc = stop recording, 'a' = align to the start pose).

    Returns an object exposing ``start()`` / ``stop()`` (mirroring pynput's Listener),
    or ``None`` when stdin is not an interactive TTY or termios is unavailable (e.g.
    Windows or a piped/non-interactive stdin), preserving the previous no-keyboard
    behaviour in those cases.
    """
    import sys

    try:
        import atexit
        import os
        import select
        import termios
        import threading
        import tty
    except Exception:
        return None

    if not sys.stdin.isatty():
        return None

    fd = sys.stdin.fileno()

    class _StdinKeyListener:
        def __init__(self):
            self._old_term = None
            self._thread = None
            self._stop_evt = threading.Event()

        def start(self):
            self._old_term = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            atexit.register(self.stop)
            self._thread = threading.Thread(
                target=self._run, name="stdin-key-listener", daemon=True
            )
            self._thread.start()

        def _run(self):
            try:
                while not self._stop_evt.is_set():
                    ready, _, _ = select.select([fd], [], [], 0.1)
                    if not ready:
                        continue
                    ch = os.read(fd, 1)
                    if ch != b"\x1b":
                        # Plain (non-escape) keys: only 'a' is bound, to request a
                        # start-pose alignment while the recording gate is waiting.
                        if ch in (b"a", b"A"):
                            print("'a' key pressed. Aligning the arm to the start pose...")
                            events["align_start_pose"] = True
                        continue
                    seq = b"\x1b"
                    # Grab the rest of an escape sequence if it arrived in the same burst.
                    # Over ssh the ESC byte and the rest of an arrow sequence ([C etc.) can arrive in
                    # separate packets more than 20 ms apart, and the arrow was then silently dropped.
                    # 100 ms covers that; a bare ESC (stop) is only confirmed 100 ms later.
                    more, _, _ = select.select([fd], [], [], 0.10)
                    if more:
                        seq += os.read(fd, 8)
                    action = _decode_control_key(seq)
                    if action == "next":
                        print("Right arrow key pressed. Exiting loop...")
                        events["exit_early"] = True
                    elif action == "rerecord":
                        print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                        events["rerecord_episode"] = True
                        events["exit_early"] = True
                    elif action == "stop":
                        print("Escape key pressed. Stopping data recording...")
                        events["stop_recording"] = True
                        events["exit_early"] = True
            except Exception as e:
                print(f"Error handling key press: {e}")

        def stop(self):
            self._stop_evt.set()
            if self._old_term is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, self._old_term)
                except Exception:
                    pass
                self._old_term = None

    listener = _StdinKeyListener()
    listener.start()
    return listener


def init_keyboard_listener():
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (right arrow, left arrow, escape, 'a') to
    control the program flow during execution, such as stopping recording, exiting loops, or
    requesting a start-pose alignment while the recording start-pose gate is waiting.

    When a graphical display is available it uses ``pynput``. On Linux without a display
    (e.g. a plain SSH session) pynput cannot receive key events, so it falls back to reading
    the same control keys from the controlling TTY via termios, keeping headless data
    collection controllable. If no interactive TTY is available either, keyboard input is
    disabled (as before).

    Returns:
        A tuple containing:
        - A listener instance exposing ``start()`` / ``stop()`` (pynput's ``Listener`` or the
          stdin fallback), or ``None`` if no keyboard input is possible.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["align_start_pose"] = False

    import os
    import sys

    # pynput's global key listener needs a graphical display (X/Wayland) on Linux and
    # therefore receives nothing over a plain SSH session, even when `import pynput`
    # succeeds (so `is_headless()` returns False). Use pynput only when a display is
    # actually available; otherwise fall back to reading control keys from the TTY.
    use_pynput = not is_headless()
    if sys.platform not in ("darwin", "win32"):
        use_pynput = use_pynput and bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )

    if not use_pynput:
        listener = _init_stdin_key_listener(events)
        if listener is None:
            logging.warning(
                "Headless environment with no interactive TTY detected. On-screen cameras "
                "display and keyboard inputs will not be available."
            )
        else:
            logging.info(
                "No display detected: reading recording control keys from stdin "
                "(right arrow = next, left arrow = rerecord, esc = stop, a = align to start pose). "
                "On-screen camera display will not be available."
            )
        return listener, events

    # Only import pynput when a display is available
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
            elif key == keyboard.KeyCode.from_char("a"):
                print("'a' key pressed. Aligning the arm to the start pose...")
                events["align_start_pose"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    require_package("deepdiff", extra="deepdiff-dep")

    from lerobot.utils.constants import DEFAULT_FEATURES

    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )


########################################################################################
# Teleoperator smooth handover helpers
# NOTE(Maxime): These functions use minimal type hints to maintain compatibility with utils
# being a root module.
########################################################################################


def teleop_supports_feedback(teleop) -> bool:
    """Return True when the teleop can receive position feedback (is actuated).

    Actuated teleops (e.g. SO-101, OpenArmMini) have non-empty ``feedback_features``
    and expose ``enable_torque`` / ``disable_torque`` motor-control methods.

    TODO(Maxime): See if it is possible to unify this interface across teleops instead of duck-typing.
    """
    return (
        bool(teleop.feedback_features)
        and hasattr(teleop, "disable_torque")
        and hasattr(teleop, "enable_torque")
    )


def teleop_smooth_move_to(teleop, target_pos: dict, duration_s: float = 2.0, fps: int = 30) -> None:
    """Smoothly move an actuated teleop to ``target_pos`` via linear interpolation.

    Requires the teleoperator to support feedback (i.e. have non-empty
    ``feedback_features`` and implement ``disable_torque`` / ``enable_torque``).

    ``target_pos`` is expected to be in the teleop's action/feedback key space.
    For homogeneous setups (e.g. SO-101 leader + SO-101 follower) this matches
    the robot action key space directly.

    TODO(Maxime): This blocks up to ``duration_s`` seconds; during this time the
    follower robot does not receive new actions, which could be an issue on LeKiwi.
    """
    teleop.enable_torque()
    current = teleop.get_action()
    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {
            k: current[k] * (1 - t) + target_pos[k] * t if k in target_pos else current[k] for k in current
        }
        teleop.send_feedback(interp)
        time.sleep(1 / fps)


def teleop_can_be_driven(teleop) -> bool:
    """Return True when the teleop may be driven to a target pose by software.

    Stricter than :func:`teleop_supports_feedback`: a teleop configured for
    torque-off manual control (e.g. OpenArm leader with ``manual_control=True``)
    is mechanically drivable but must not be driven, because enabling torque to
    move it leaves it stiff afterwards and the operator can no longer move it by
    hand. Gravity-compensation mode is fine — the teleop loop re-injects the
    gravity feed-forward on the next ``get_action()`` and the arm goes weightless
    again on its own.
    """
    if not teleop_supports_feedback(teleop):
        return False
    return not getattr(teleop.config, "manual_control", False)


def _banner(title: str, keys: str) -> None:
    """Operator-facing state banner: plain print so it stands out from the INFO stream
    (the per-bus Damiao stats lines bury a one-line prompt within seconds)."""
    line = "=" * 72
    print(f"\n{line}\n  {title}\n  {keys}\n{line}\n", flush=True)


def start_pose_deviation(current_pos: dict, target_pos: dict) -> dict[str, float]:
    """Absolute per-joint deviation (deg) between the current pose and a start pose.

    Both dicts are in the observation/action key space (``<motor>.pos``). Keys
    absent from either side are skipped, so a partial ``target_pos`` gates only
    on the joints it names.
    """
    return {
        k: abs(float(current_pos[k]) - float(target_pos[k]))
        for k in target_pos
        if k in current_pos and current_pos[k] is not None and target_pos[k] is not None
    }


def wait_for_start_pose(
    robot,
    events: dict,
    target_pos: dict,
    tolerance_deg: float,
    teleop=None,
    teleop_slice=None,
    verify_pos: dict | None = None,
    slice_s: float = 2.0,
    max_wait_s: float = 120.0,
    play_sounds: bool = True,
) -> tuple[dict[str, float], str]:
    """Block until the robot's pose is within ``tolerance_deg`` of ``target_pos``.

    Data-collection quality gate. The startup alignment only fires once per
    session, so with a manual per-episode reset every episode starts from a
    slightly different pose — measured on OpenArm, 19-28% of the joint range on
    J1/J4/J6/J7. Trained on few episodes, a policy then regresses toward the mean
    start pose instead of learning the task, so this gate refuses to begin an
    episode until the arm is back at the pose captured at connect time.

    The arm is never moved on its own: pressing 'a' drives the leader to
    ``target_pos`` (only when :func:`teleop_can_be_driven`), otherwise the
    operator moves it by hand. ``teleop_slice`` runs ``slice_s`` seconds of the
    normal teleop loop between checks so the follower keeps tracking the leader
    while the operator adjusts — without it the follower would not move at all
    while the gate waits.

    Returns ``(deviation, outcome)`` with outcome one of ``"within"`` (passed),
    ``"override"`` (operator skipped it with the right arrow), ``"timeout"``, or
    ``"stopped"`` (esc).
    """
    from lerobot.utils.utils import play_cue

    # ``target_pos`` is what the arm is driven to (every joint, gripper included);
    # ``verify_pos`` is the subset the gate judges, so a joint whose value varies
    # legitimately between episodes can be returned without being gated on.
    checked = verify_pos if verify_pos is not None else target_pos

    deadline = time.perf_counter() + max_wait_s
    announced = False

    if events.get("align_start_pose"):
        # Pressed before the gate opened (e.g. during the reset window, where
        # nothing consumes it). Honouring it here would start an autonomous move
        # seconds after the keypress, while the operator may still be reaching
        # into the workspace. Require a fresh press instead.
        events["align_start_pose"] = False
        logging.info(
            "Start-pose gate: discarding an alignment request made before the gate opened; "
            "press 'a' again once clear of both arms."
        )

    while True:
        if events.get("rerecord_episode"):
            # No episode is in progress while the gate waits, so a rerecord
            # request has nothing to act on. Drop it here: left arrow also sets
            # exit_early, and leaving the flag set would make the caller record
            # the next episode and then discard it.
            events["rerecord_episode"] = False
            logging.info("Start-pose gate: ignoring rerecord request (no episode in progress).")

        obs = robot.get_observation()
        current_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
        deviation = start_pose_deviation(current_pos, checked)

        if not deviation:
            logging.warning(
                "Start-pose gate: no joints in common between the observation and the start pose; skipping."
            )
            return deviation, "within"

        worst = max(deviation, key=deviation.get)
        if deviation[worst] <= tolerance_deg:
            if announced:
                logging.info(
                    "Start-pose gate cleared (worst %s=%.1fdeg <= %.1fdeg).",
                    worst,
                    deviation[worst],
                    tolerance_deg,
                )
            return deviation, "within"

        if events.get("stop_recording"):
            return deviation, "stopped"

        if events.get("exit_early"):
            # Right arrow while the gate waits = record from here anyway.
            events["exit_early"] = False
            logging.warning(
                "Start-pose gate overridden by the operator (worst %s=%.1fdeg > %.1fdeg).",
                worst,
                deviation[worst],
                tolerance_deg,
            )
            return deviation, "override"

        if not announced:
            play_cue("message", play_sounds)
            announced = True

        offenders = ", ".join(
            f"{k}={deviation[k]:.1f}deg"
            for k in sorted(deviation, key=deviation.get, reverse=True)
            if deviation[k] > tolerance_deg
        )
        logging.info(
            "Start-pose gate: waiting (tolerance %.1fdeg). Off by %s. "
            "Press 'a' to align, right arrow to record anyway, esc to stop.",
            tolerance_deg,
            offenders,
        )

        if events.get("align_start_pose"):
            events["align_start_pose"] = False
            drive_to_start_pose(robot, teleop, target_pos)

        if time.perf_counter() > deadline:
            logging.warning(
                "Start-pose gate timed out after %.0fs (worst %s=%.1fdeg > %.1fdeg). Recording anyway.",
                max_wait_s,
                worst,
                deviation[worst],
                tolerance_deg,
            )
            return deviation, "timeout"

        if teleop_slice is not None:
            # Run the real teleop loop so the follower tracks the leader as the
            # operator (or the alignment above) brings it back to the start pose.
            teleop_slice(slice_s)
        else:
            time.sleep(min(slice_s, 0.5))


def drive_to_start_pose(robot, teleop, target_pos: dict) -> bool:
    """Move both arms back to ``target_pos``, leader first, then follower.

    The order matters: while teleoperating, the leader's pose commands the
    follower, so moving the follower alone is undone by the next tracking cycle.
    Walking the follower over afterwards is what keeps the hand-off gentle —
    otherwise the first tracking cycle commands it to the leader's new pose in a
    single step, and nothing rate-limits that path unless ``max_relative_target``
    is set (it defaults to None).

    Torque is deliberately left on afterwards: a gravity-compensated leader goes
    weightless again on its next ``get_action()``, whereas disabling torque here
    would drop the arm under its own weight.

    Returns False when the teleop must not be driven by software (a torque-off
    manual-control leader), in which case nothing is moved.
    """
    if teleop is None:
        logging.warning("Start pose: no teleoperator to drive; move the arm by hand instead.")
        return False

    if hasattr(teleop, "soft_move_to"):
        # Teleops that drive themselves on their own bus (OpenArm leader: MIT
        # position commands with gravity feed-forward, the same primitive as its
        # connect-time alignment). The generic path below goes through
        # send_feedback, which OpenArm raises NotImplementedError for, so asking
        # the teleop to move itself is the only way it can be returned at all.
        logging.info("Start pose: driving the leader back to the start pose.")
        if not teleop.soft_move_to(
            {k.removesuffix(".pos"): v for k, v in target_pos.items()}, duration_s=2.0
        ):
            return False
    elif teleop_can_be_driven(teleop):
        logging.info("Start pose: driving the leader back to the start pose.")
        teleop_smooth_move_to(teleop, target_pos, duration_s=2.0)
    else:
        logging.warning(
            "Start pose: this teleop must not be driven by software; move the arm by hand instead."
        )
        return False

    current_pos = {k: v for k, v in robot.get_observation().items() if k in target_pos}
    if current_pos:
        logging.info("Start pose: walking the follower over.")
        follower_smooth_move_to(robot, current_pos, target_pos, duration_s=1.5)
    return True


def teleop_pose_deviation(teleop, target_pos: dict) -> dict[str, float]:
    """Per-joint deviation (deg) of the teleop itself from ``target_pos``, read without releasing it.

    While the arms are held at the start pose the leader is decoupled from the follower, so the
    follower's own deviation says nothing about where the leader is. ``get_action()`` cannot be
    used to look: on a gravity-compensated leader it re-injects the gravity torque with zero
    gains, which is exactly what ends the hold. Teleops that can report their joints with a
    torque-neutral read expose ``read_positions()``; for any other teleop this returns ``{}``
    and the caller has nothing to judge on.
    """
    read = getattr(teleop, "read_positions", None)
    if read is None:
        return {}
    return start_pose_deviation(read(), target_pos)


def held_teleop_ready(teleop, checked_pos: dict, max_deg: float) -> bool:
    """Whether a held teleop is close enough to the start pose to hand the follower back to it.

    The follower stays at the start pose for as long as the arms are held, whatever happens to
    the leader. The first tracking cycle of the next episode then commands the follower to the
    leader's pose in a single step, and nothing rate-limits that path unless
    ``max_relative_target`` is set. A leader that was pushed away, or that lost its hold, has to
    be returned before recording starts.
    """
    deviation = teleop_pose_deviation(teleop, checked_pos)
    if not deviation:
        return True
    worst = max(deviation, key=deviation.get)
    logging.info(
        "Start pose hand-off: leader worst %s=%.1fdeg (limit %.1fdeg).", worst, deviation[worst], max_deg
    )
    if deviation[worst] <= max_deg:
        return True
    logging.warning(
        "Start pose hand-off refused: the leader is off the start pose (worst %s=%.1fdeg > %.1fdeg). "
        "Press 'a' to return both arms again.",
        worst,
        deviation[worst],
        max_deg,
    )
    return False


def countdown_to_episode_start(events: dict, seconds: int, run_slice=None) -> bool:
    """Count down on the console before the first recorded frame. Returns False when esc was pressed.

    Without it the right arrow gave no usable cue: recording began the moment the previous take
    finished saving, a few seconds later and with only a banner to say so. One line and one
    terminal bell per second give the operator a fixed lead-in to get both hands onto the leader.

    ``run_slice(seconds)`` is what runs during each second of the count: the teleop loop when the
    follower should keep tracking the leader, or an observe-only loop while the arms are held at
    the start pose, where they have to stay put until the first frame. None just waits. Keys
    other than esc pressed during the count are dropped: a second right arrow would otherwise end
    the episode on its first frame.
    """
    if seconds <= 0:
        return not events.get("stop_recording")
    for remaining in range(math.ceil(seconds), 0, -1):
        if events.get("stop_recording"):
            return False
        print(f"\a  録画開始まで {remaining} …", flush=True)
        if run_slice is not None:
            run_slice(1.0)
        else:
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline and not events.get("stop_recording"):
                time.sleep(0.05)
    if events.get("stop_recording"):
        return False
    for key in ("exit_early", "rerecord_episode", "align_start_pose"):
        events[key] = False
    return True


def wait_for_episode_cue(
    robot,
    events: dict,
    target_pos: dict,
    teleop=None,
    teleop_slice=None,
    verify_deg: float | None = None,
    verify_pos: dict | None = None,
    slice_s: float = 2.0,
    reprompt_s: float = 15.0,
    play_sounds: bool = True,
    hold: bool = False,
    initial_state: str = "free",
    handoff_deg: float | None = None,
    discardable: bool = True,
    on_returned: Callable[[], bool] | None = None,
    hold_slice: Callable[[float], None] | None = None,
    hold_watch_s: float = 1.0,
) -> tuple[dict[str, float], str, str]:
    """Hand control of the episode boundary to the operator (manual advance).

    Implements the collection flow:

    1. the episode finishes,
    2. the operator presses 'a' and both arms are driven back to the start pose,
       clearing the workspace before anything is placed in it,
    3. the operator restores the scene,
    4. the operator presses the right arrow and the next episode starts.

    Returning the arm *before* the scene is restored is deliberate: the return is
    a joint-space interpolation with no collision checking, so doing it after the
    object is placed risks sweeping the object off its start position.

    The wait is unbounded. Teleoperated collection is attended by definition, and
    auto-advancing here would record exactly the misaligned episode this exists to
    prevent. ``teleop_slice`` runs the normal teleop loop between polls, so the
    follower keeps tracking the leader and the operator can still move the arm by
    hand while a cue is pending.

    Pressing the right arrow without having pressed 'a' still starts the episode
    — the operator keeps the last word — but the outcome records that the arm was
    never returned.

    ``target_pos`` is what the arms are returned to and should cover every joint,
    the gripper included, so collection starts from the same full state that
    lerobot-rollout returns to between inference episodes. ``verify_pos`` is the
    subset the result is checked against, defaulting to ``target_pos``; the caller
    can drop joints whose end-of-episode value legitimately varies (the gripper,
    which depends on what was grasped) from the check without excluding them from
    the return itself.

    With ``hold``, a verified return stops the teleop loop instead of resuming it:
    the leader keeps the soft position hold its return ended in, the follower keeps
    its last command, and both stay at the start pose until the caller records the
    first frame. A gravity-compensated leader has no restoring stiffness on the
    wrist roll; left to the teleop loop it rolls tens of degrees while the scene is
    being restored, the follower tracks it into whatever is nearby, and the stalled
    joint overheats. While held, both arms are only read (every ``hold_watch_s``),
    which keeps their feedback on the bus and the per-bus drop warnings alive, and
    ``hold_slice`` stands in for ``teleop_slice`` so the caller can keep observing
    (and displaying) without commanding anything.

    The wait is in one of three states, returned so the next call can pick up
    where this one left the arms (``initial_state``):

    - ``"free"``: the teleop loop runs (before 'a', or without ``hold``).
    - ``"held"``: returned and held at the start pose.
    - ``"blocked"``: held, but the leader is more than ``handoff_deg`` off the
      start pose. Releasing it would step the follower to the leader's pose, so
      the right arrow is refused until 'a' has returned the arms.

    ``on_returned`` runs once after the first verified return and reports whether
    it committed the pending take (the caller saves it there, so the save overlaps
    with restoring the scene instead of delaying the start). A committed take can
    no longer be discarded, and neither can anything when ``discardable`` is False
    (no take is pending): the left arrow is then dropped instead of being passed
    on, where it used to fall through into starting a recording.

    Returns ``(deviation, outcome, state)`` with outcome one of ``"started"``
    (returned, verified, then started), ``"started_unhomed"`` (started without a
    verified return), ``"rerecord"`` (left arrow), or ``"stopped"`` (esc).
    """
    from lerobot.utils.utils import play_cue

    if initial_state not in ("free", "held", "blocked"):
        raise ValueError(f"initial_state must be 'free', 'held' or 'blocked', got {initial_state!r}")

    checked = verify_pos if verify_pos is not None else target_pos

    def measure() -> dict[str, float]:
        obs = robot.get_observation()
        current = {k: v for k, v in obs.items() if k.endswith(".pos")}
        return start_pose_deviation(current, checked)

    # ``paused``: the teleop loop is not run, so neither arm receives a new command.
    paused = hold and initial_state != "free"
    homed = hold and initial_state == "held"
    committed = False
    prompted_at: float | None = None
    prompted_state: str | None = None
    watched_at = 0.0
    warned_at: float | None = None

    def current_state() -> str:
        if paused:
            return "held" if homed else "blocked"
        return "free"

    while True:
        if events.get("stop_recording"):
            return measure(), "stopped", current_state()

        if events.get("rerecord_episode"):
            if discardable and not committed:
                # Left arrow: the caller discards the episode just recorded. Leave the
                # flag set so the existing rerecord handling runs unchanged.
                return measure(), "rerecord", current_state()
            # Left arrow also sets exit_early; left set, it would start the episode.
            events["rerecord_episode"] = False
            events["exit_early"] = False
            logging.info(
                "Left arrow ignored: %s.",
                "the take is already saved" if committed else "no take is pending",
            )
            _banner(
                "破棄できるテイクが無い（" + ("直前のテイクは保存済み" if committed else "録画前") + "）",
                "← は無視した",
            )
            prompted_at = None
            continue

        if events.get("align_start_pose"):
            events["align_start_pose"] = False
            moved = drive_to_start_pose(robot, teleop, target_pos)
            deviation = measure()
            worst = max(deviation, key=deviation.get) if deviation else None
            if worst is not None and verify_deg is not None and deviation[worst] > verify_deg:
                # The return did not land: a dropped CAN packet, the arm stopping
                # against an obstacle, or sag at a lifted pose. Say so instead of
                # letting a silently-misaligned episode be recorded.
                logging.warning(
                    "Start pose: the return did not reach the start pose (worst %s=%.1fdeg > %.1fdeg). "
                    "Press 'a' again, or the right arrow to record from here anyway.",
                    worst,
                    deviation[worst],
                    verify_deg,
                )
                # Not held either: a follower stopped short is pressing on something, and
                # the teleop loop is how the operator moves it clear.
                homed = False
                paused = False
                _banner(
                    f"戻し損ね: {worst} が {deviation[worst]:.1f}° ずれている（録画していない）",
                    "a もう一度戻す   → ここから録画を強行   esc 終了",
                )
                prompted_state = "needs_home"
            elif moved or worst is None:
                homed = True
                # Only a teleop that was actually driven is left holding its pose.
                paused = hold and moved
                play_cue("complete", play_sounds)
                if on_returned is not None and not committed:
                    committed = bool(on_returned())
                prompted_state = None
            prompted_at = time.perf_counter()
            continue

        if events.get("exit_early"):
            events["exit_early"] = False
            if (
                paused
                and homed
                and handoff_deg is not None
                and not held_teleop_ready(teleop, checked, handoff_deg)
            ):
                homed = False
            if paused and not homed:
                prompted_at = None
                continue
            deviation = measure()
            if homed:
                return deviation, "started", current_state()
            worst = max(deviation, key=deviation.get) if deviation else None
            if worst is not None:
                logging.warning(
                    "Starting the episode without returning the arm (worst %s=%.1fdeg).",
                    worst,
                    deviation[worst],
                )
            return deviation, "started_unhomed", current_state()

        state = current_state()
        if state == "free":
            state = "homed" if homed else "needs_home"
        now = time.perf_counter()
        if prompted_state != state or prompted_at is None or now - prompted_at >= reprompt_s:
            discard_key = "   ← 直前のテイクを破棄" if discardable and not committed else ""
            saved_note = "（前のテイクは保存済み）" if committed else ""
            if state == "needs_home":
                if prompted_state != state:
                    play_cue("message", play_sounds)
                logging.info(
                    "Waiting: press 'a' to return both arms to the start pose, "
                    "right arrow to start without returning, left arrow to re-record, esc to stop."
                )
                _banner(
                    "待機中（録画していない）",
                    f"→ 開始   a 両腕を開始姿勢へ戻す（手を引く）{discard_key}   esc 終了",
                )
            elif state == "blocked":
                logging.info(
                    "Waiting: the leader is off the start pose while held. Press 'a' to return both arms "
                    "(esc stops)."
                )
                _banner(
                    "保持中: leader が開始姿勢からずれている（録画していない・→ は効かない）",
                    "a もう一度戻す（手を引く）   esc 終了",
                )
            elif state == "held":
                logging.info(
                    "Waiting: both arms are held at the start pose. Restore the scene, then press the "
                    "right arrow to start ('a' returns them again, esc stops)."
                )
                _banner(
                    f"待機中: 両腕を開始姿勢で保持している（録画していない）{saved_note}",
                    f"シーンを整えて → で開始   a もう一度戻す{discard_key}   esc 終了",
                )
            else:
                logging.info(
                    "Waiting: arm is at the start pose. Restore the scene, then press the right arrow "
                    "to start ('a' returns it again, esc stops)."
                )
                _banner(
                    f"待機中: 腕は開始姿勢に戻っている（録画していない）{saved_note}",
                    "シーンを整えて → で開始"
                    + ("" if committed else "（前のテイクが確定保存）")
                    + "   a もう一度戻す   esc 終了",
                )
            prompted_at = now
            prompted_state = state

        if paused:
            if now - watched_at >= hold_watch_s:
                watched_at = now
                # Read both arms without commanding either. A follower that has left the start
                # pose is being pushed by something; a leader that has left it lost its hold.
                off = []
                follower_dev = measure()
                if follower_dev and verify_deg is not None:
                    worst = max(follower_dev, key=follower_dev.get)
                    if follower_dev[worst] > verify_deg:
                        off.append(f"follower {worst}={follower_dev[worst]:.1f}deg")
                leader_dev = teleop_pose_deviation(teleop, checked)
                if leader_dev and handoff_deg is not None:
                    worst = max(leader_dev, key=leader_dev.get)
                    if leader_dev[worst] > handoff_deg:
                        off.append(f"leader {worst}={leader_dev[worst]:.1f}deg")
                if homed and off and (warned_at is None or now - warned_at >= 10.0):
                    warned_at = now
                    logging.warning(
                        "Start pose hold: %s off the start pose. Press 'a' to return both arms again.",
                        ", ".join(off),
                    )
            if hold_slice is not None:
                hold_slice(0.1)
            else:
                time.sleep(0.05)
        elif teleop_slice is not None:
            teleop_slice(slice_s)
        else:
            time.sleep(min(slice_s, 0.5))


def follower_smooth_move_to(
    robot, current: dict, target: dict, duration_s: float = 1.0, fps: int = 30
) -> None:
    """Smoothly move the follower robot from ``current`` to ``target`` action.

    Used when the teleop is non-actuated: instead of driving the leader arm to
    the follower, the follower is brought to the teleop's current pose so the
    robot meets the operator's hand rather than jumping to it on the first frame.

    Both ``current`` and ``target`` must be in the robot action key space
    (i.e. the output of ``robot_action_processor``).
    """
    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {k: current[k] * (1 - t) + target[k] * t if k in target else current[k] for k in current}
        robot.send_action(interp)
        time.sleep(1 / fps)
