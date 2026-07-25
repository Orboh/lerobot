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

"""
Records a dataset via teleoperation.  This is a pure data-collection
tool — no policy inference.  For deploying trained policies, use
``lerobot-rollout`` instead.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Example:

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

Example recording with bimanual so100:
```shell
lerobot-record \\
  --robot.type=bi_so_follower \\
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \\
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \\
  --robot.id=bimanual_follower \\
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \\
  --teleop.type=bi_so_leader \\
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \\
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \\
  --teleop.id=bimanual_leader \\
  --display_data=true \\
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \\
  --dataset.num_episodes=25 \\
  --dataset.single_task="Grab and handover the red cube to the other arm" \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=2
```

Example recording with custom video encoding parameters:
```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --dataset.camera_encoder.vcodec=h264 \\
    --dataset.camera_encoder.preset=fast \\
    --dataset.camera_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2} \\
    --display_data=true
```
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
    wait_for_episode_cue,
    wait_for_start_pose,
)
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Teleoperator to control the robot (required)
    teleop: TeleoperatorConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    # How far a joint may sit from the start pose, in degrees. None (default)
    # leaves the start pose unmanaged, which is how recording behaved before.
    # The reference pose is the one read right after connect, i.e. after the
    # robot's own startup alignment. What the number gates depends on
    # episode_advance: with "manual" it verifies the operator-cued return landed;
    # with "auto" it is the threshold an episode must satisfy before it starts,
    # with the operator bringing the arm back by hand. It matters because the
    # startup alignment only fires once per session, so a per-episode manual
    # reset makes every episode start from a slightly different pose (measured on
    # OpenArm: a fifth to a quarter of the joint range on J1/J4/J6/J7), and a
    # policy trained on few episodes regresses toward that mean pose.
    # Keep literal percent signs out of these comments: draccus turns them into
    # argparse help strings, and a stray percent sign there breaks --help.
    start_pose_tolerance_deg: float | None = None
    # Only used when episode_advance="auto": stop waiting for the start pose after
    # this long and record anyway, so an unattended session cannot block forever.
    # The "manual" flow waits indefinitely instead, since a human is always there.
    start_pose_max_wait_s: float = 120.0
    # Include the gripper in the start-pose check. Off by default: what it reads
    # at the end of an episode legitimately depends on what was grasped, so
    # checking it produces noise. This only affects the check — the gripper is
    # always part of what the arm is returned to, matching what lerobot-rollout
    # restores between inference episodes.
    start_pose_gate_gripper: bool = False
    # Append one JSON line per episode with the start-pose deviation to this file
    # (the collection manifest). None disables the file; deviations are logged to
    # the console either way.
    start_pose_log_path: str | None = None
    # How the recording advances between episodes.
    #   "auto"   - the timed flow: a fixed reset window, then the next episode
    #              starts on its own. Good for batching through episodes quickly.
    #   "manual" - the operator drives the boundary: 'a' returns both arms to the
    #              start pose (clearing the workspace first), then the right arrow
    #              starts the next episode once the scene is restored.
    episode_advance: str = "auto"

    def __post_init__(self):
        if self.episode_advance not in ("auto", "manual"):
            raise ValueError(f"episode_advance must be 'auto' or 'manual', got {self.episode_advance!r}")
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     [ Teleoperator ]
     |
     |  [teleop.get_action] -> raw_action
     |          |
     |          V
     | [teleop_action_processor]
     |          |
     '---> processed_teleop_action
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


def _log_start_pose_deviation(
    path: str | None,
    episode_index: int,
    deviation: dict[str, float],
    outcome: str,
    tolerance_deg: float | None,
) -> None:
    """Record how far the arm was from the start pose when an episode began.

    Always logs a one-line summary; also appends a JSON line to ``path`` when
    given, so the per-episode start pose can be audited after the fact instead of
    being reconstructed from the recorded frames (which is how the 19-28% start
    scatter went unnoticed across 25 episodes).
    """
    if not deviation:
        return

    worst = max(deviation, key=deviation.get)
    logging.info(
        "Episode %d start pose: worst %s=%.1fdeg (tolerance %s, outcome=%s).",
        episode_index,
        worst,
        deviation[worst],
        f"{tolerance_deg:.1f}deg" if tolerance_deg is not None else "not checked",
        outcome,
    )
    if not path:
        return

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episode_index": episode_index,
        "outcome": outcome,
        "tolerance_deg": tolerance_deg,
        "worst_joint": worst,
        "worst_deviation_deg": round(deviation[worst], 3),
        "deviation_deg": {k: round(v, 3) for k, v in sorted(deviation.items())},
    }
    try:
        log_path = Path(path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # A failed audit log must never abort a recording session.
        logging.warning("Could not append the start-pose log to %s: %s", path, e)


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    control_interval = 1 / fps

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from teleop
        if isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        elif isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t

        sleep_time_s: float = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
            )

        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Fall back to identity pipelines when the caller doesn't supply processors.
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Reject eval_ prefix — for policy evaluation use lerobot-rollout
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.camera_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        # The pose right after connect is the start pose: the robot's startup
        # alignment has just run, so this is the "correct" pose every episode
        # should begin from. Captured once and reused as the target every episode
        # returns to — the same thing lerobot-rollout does via its initial_position.
        start_pose: dict[str, float] = {}
        verify_pose: dict[str, float] = {}
        if cfg.start_pose_tolerance_deg is not None or cfg.episode_advance == "manual":
            # Every joint, gripper included: this is what the arms are returned to,
            # and it has to match what lerobot-rollout returns to between inference
            # episodes (which covers the full .pos set) or collection and inference
            # would start from different states.
            start_pose = {
                k: v for k, v in robot.get_observation().items() if k.endswith(".pos") and v is not None
            }
            # The gripper is returned but not judged by default: what it reads at
            # the end of an episode legitimately depends on what was grasped.
            verify_pose = {
                k: v for k, v in start_pose.items() if cfg.start_pose_gate_gripper or "gripper" not in k
            }
            if start_pose:
                logging.info(
                    "Start pose captured on %d joints, %d checked (advance=%s, tolerance=%s): %s",
                    len(start_pose),
                    len(verify_pose),
                    cfg.episode_advance,
                    f"{cfg.start_pose_tolerance_deg:.1f}deg"
                    if cfg.start_pose_tolerance_deg is not None
                    else "not checked",
                    ", ".join(f"{k}={v:.1f}" for k, v in sorted(start_pose.items())),
                )
                if cfg.episode_advance == "manual":
                    logging.info(
                        "Manual episode advance: after each episode press 'a' to return both arms to "
                        "the start pose, restore the scene, then press the right arrow to start the "
                        "next episode (left arrow re-records, esc stops)."
                    )
            else:
                logging.warning(
                    "Start-pose handling requested but the robot reported no usable joint positions; "
                    "recording without it."
                )

        def start_pose_teleop_slice(seconds: float) -> None:
            """Run the plain teleop loop briefly so the follower tracks the leader.

            Uses a private events dict: ``record_loop`` consumes ``exit_early``
            when it breaks, which would swallow the operator's right-arrow
            override before the gate could see it on the shared dict.
            """
            record_loop(
                robot=robot,
                events={
                    "exit_early": False,
                    "rerecord_episode": False,
                    "stop_recording": False,
                    "align_start_pose": False,
                },
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=seconds,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            # Start-pose result carried over from the manual cue, logged at the top
            # of the next iteration where the episode index it belongs to is known
            # (a re-record reuses the same index).
            pending_start_pose: tuple[dict[str, float], str] | None = None
            # Set after a discarded take in manual mode: the retake needs the arm
            # returned again, and the cue for it has to run before recording.
            cue_before_next_record = False
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                if pending_start_pose is not None:
                    _log_start_pose_deviation(
                        cfg.start_pose_log_path,
                        dataset.num_episodes,
                        *pending_start_pose,
                        cfg.start_pose_tolerance_deg,
                    )
                    pending_start_pose = None

                if cue_before_next_record and start_pose:
                    cue_before_next_record = False
                    deviation, outcome = wait_for_episode_cue(
                        robot=robot,
                        events=events,
                        target_pos=start_pose,
                        verify_pos=verify_pose,
                        teleop=teleop,
                        teleop_slice=start_pose_teleop_slice,
                        verify_deg=cfg.start_pose_tolerance_deg,
                        play_sounds=cfg.play_sounds,
                    )
                    _log_start_pose_deviation(
                        cfg.start_pose_log_path,
                        dataset.num_episodes,
                        deviation,
                        outcome,
                        cfg.start_pose_tolerance_deg,
                    )
                    if outcome == "stopped":
                        break
                    if outcome == "rerecord":
                        # Nothing has been recorded yet to discard.
                        events["rerecord_episode"] = False

                if start_pose and cfg.episode_advance == "auto":
                    deviation, outcome = wait_for_start_pose(
                        robot=robot,
                        events=events,
                        target_pos=start_pose,
                        verify_pos=verify_pose,
                        tolerance_deg=cfg.start_pose_tolerance_deg,
                        teleop=teleop,
                        teleop_slice=start_pose_teleop_slice,
                        max_wait_s=cfg.start_pose_max_wait_s,
                        play_sounds=cfg.play_sounds,
                    )
                    _log_start_pose_deviation(
                        cfg.start_pose_log_path,
                        dataset.num_episodes,
                        deviation,
                        outcome,
                        cfg.start_pose_tolerance_deg,
                    )
                    if outcome == "stopped":
                        break

                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                is_last_episode = recorded_episodes >= cfg.dataset.num_episodes - 1
                if cfg.episode_advance == "manual" and start_pose:
                    # The operator owns the boundary: 'a' returns the arms (which
                    # clears the workspace before anything is placed in it), then
                    # the right arrow starts the next episode. Skipped after the
                    # final episode, where there is no next one to set up.
                    if not events["stop_recording"] and (not is_last_episode or events["rerecord_episode"]):
                        pending_start_pose = wait_for_episode_cue(
                            robot=robot,
                            events=events,
                            target_pos=start_pose,
                            verify_pos=verify_pose,
                            teleop=teleop,
                            teleop_slice=start_pose_teleop_slice,
                            verify_deg=cfg.start_pose_tolerance_deg,
                            play_sounds=cfg.play_sounds,
                        )
                        if pending_start_pose[1] == "stopped":
                            # Fall through so the episode just recorded is still
                            # saved; the while condition ends the run.
                            pending_start_pose = None
                        elif pending_start_pose[1] == "rerecord":
                            # The handler below discards the take. The arm still
                            # has to be returned before the retake, so ask for the
                            # cue again at the top of the loop.
                            pending_start_pose = None
                            cue_before_next_record = True
                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                elif not events["stop_recording"] and (not is_last_episode or events["rerecord_episode"]):
                    log_say("Reset the environment", cfg.play_sounds)

                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
