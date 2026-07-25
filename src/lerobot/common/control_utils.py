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
import time
import traceback
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
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
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
                    more, _, _ = select.select([fd], [], [], 0.02)
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
    from lerobot.utils.utils import log_say

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
        deviation = start_pose_deviation(current_pos, target_pos)

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
            log_say("Return the arm to the start pose", play_sounds)
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
            if teleop is not None and teleop_can_be_driven(teleop):
                logging.info("Start-pose gate: driving the teleop to the start pose.")
                # Torque is deliberately left on: a gravity-compensated leader
                # goes weightless again on the next get_action(), while
                # disabling it here would drop the arm under its own weight.
                teleop_smooth_move_to(teleop, target_pos, duration_s=2.0)
                # Walk the follower over as well, in the same order the teleop
                # loop would: leader first, then follower. Without this the
                # follower sits still while the leader moves, and the first
                # tracking cycle after the gate commands it straight to the new
                # pose in one step — there is no rate limit on that path unless
                # max_relative_target is set (it defaults to None).
                current_pos = {k: v for k, v in robot.get_observation().items() if k in target_pos}
                if current_pos:
                    follower_smooth_move_to(robot, current_pos, target_pos, duration_s=1.5)
            else:
                logging.warning(
                    "Start-pose gate: this teleop must not be driven by software; move the arm by hand."
                )

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
