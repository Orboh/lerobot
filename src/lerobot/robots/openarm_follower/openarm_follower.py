#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.damiao_alignment import (
    check_start_position,
    resolve_initial_pose,
    should_rezero_on_connect,
    soft_move_to_position,
)
from lerobot.motors.damiao.damiao_bump_calibration import run_bump_to_stop_calibration
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_openarm_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
    OpenArmFollowerConfig,
)

logger = logging.getLogger(__name__)


class OpenArmFollower(Robot):
    """
    OpenArms Follower Robot which uses CAN bus communication to control 7 DOF arm with a gripper.
    The arm uses Damiao motors in MIT control mode.
    """

    config_class = OpenArmFollowerConfig
    name = "openarm_follower"

    def __init__(self, config: OpenArmFollowerConfig):
        super().__init__(config)
        if self.id is None:
            raise ValueError(
                f"{self.name} requires an explicit id "
                "(e.g. --robot.id=follower_right). Without one the calibration path "
                "collapses to 'None.json', which every unnamed OpenArm device "
                "would share regardless of side."
            )
        self.config = config

        # Arm motors
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(
                send_id, motor_type_str, MotorNormMode.DEGREES
            )  # Always use degrees for Damiao motors
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        if config.side is not None:
            if config.side == "left":
                config.joint_limits = LEFT_DEFAULT_JOINTS_LIMITS
            elif config.side == "right":
                config.joint_limits = RIGHT_DEFAULT_JOINTS_LIMITS
            else:
                raise ValueError(
                    "config.side must be either 'left', 'right' (for default values) or 'None' (for CLI values)"
                )
        else:
            logger.info(
                "Set config.side to either 'left' or 'right' to use pre-configured values for joint limits."
            )
        logger.info(f"Values used for joint limits: {config.joint_limits}.")

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot and optionally calibrate.

        We assume that at connection time, the arms are in a safe rest position,
        and torque can be safely disabled to run calibration if needed.
        """

        # Connect to CAN bus
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        # Run calibration if needed
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        # Persistent zero: decide ONCE whether to burn a fresh motor-side zero at
        # the current pose (legacy, forces the operator to hang the arm down
        # identically every session) or to trust the software homing offsets in
        # the calibration file (new flow; the frame is stable across sessions and
        # power cycles). See should_rezero_on_connect for the auto rule.
        rezero = should_rezero_on_connect(self.bus.calibration, self.config.rezero_on_connect)
        if not rezero and self.config.align_on_connect:
            # No re-zero -> the frame must already be valid. Refuse to run the
            # startup alignment move if the readings look grossly out of frame
            # (stale zero after power cycle etc.). Gated on align_on_connect so
            # a motion-free connect (e.g. lerobot-calibrate re-calibrating a
            # stale frame with align disabled) is never dead-locked by the check.
            check_start_position(
                self.bus,
                self.config.joint_limits,
                self.config.start_position_tolerance_deg,
                label=str(self),
            )

        self.configure()

        if rezero:
            self.bus.set_zero_position()

        self.bus.enable_torque()

        # Startup alignment (official AdjustPosition port): softly move to the
        # initial pose so the follower starts matched with the leader. The target
        # is the captured "ready" pose (initial_pose_path / initial_pose_deg) or
        # the official default; it is clamped into joint_limits. Placed after
        # set_zero_position so the pose targets use the fresh zero.
        if self.config.align_on_connect:
            goal = resolve_initial_pose(
                initial_pose_deg=self.config.initial_pose_deg,
                initial_pose_path=self.config.initial_pose_path,
                joint_limits=self.config.joint_limits,
            )
            soft_move_to_position(self.bus, goal, self.config.align_duration_s)

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms robot (persistent software zero).

        Two anchor modes (config.calibration_anchor):

        "hang_down" (default):
        1. Disable torque
        2. Ask user to position the arm hanging straight down, gripper closed
        3. Read the motors' NATIVE positions at that reference and store them as
           software homing offsets in the calibration file (logical zero =
           hang-down = URDF zero). The motor-side zero (Damiao 0xFE) is
           intentionally NOT burned: DM motors have a single-turn absolute
           encoder, so native readings are repeatable across power cycles, while
           0xFE flash persistence is firmware-dependent — the software offset is
           robust either way. Burn the motor zero out-of-band with the official
           tools (openarm-can-cli set_zero) only if you need the native frame
           itself re-centered, then re-run this calibration.
        4. Save calibration (with the fixed ±90° default range)

        "bump_to_stop" (opt-in): the arm actively sweeps each ARM joint into its
        mechanical hard stop (official openarm_can sequence port) and the homing
        offset is computed from the native reading at each stop and the known
        stop angle (offset = raw_at_stop - stop_angle) — no dependence on how
        the operator positioned the arm. The gripper is never bumped: its zero
        must be flash-burned by hand at the closed pose first
        (openarm_gripper_zero.py) and its offset is pinned to 0. See
        lerobot.motors.damiao.damiao_bump_calibration for safety notes and
        per-unit tuning (thresholds / stop angles are config-overridable).

        Once a calibration with homing offsets exists, connect() stops re-zeroing
        every session (see rezero_on_connect) — the start frame is fixed and
        reproducible without hanging the arm down at each startup.
        """
        if self.config.calibration_anchor not in ("hang_down", "bump_to_stop"):
            raise ValueError(
                f"calibration_anchor must be 'hang_down' or 'bump_to_stop', "
                f"got {self.config.calibration_anchor!r}"
            )
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")

        if self.config.calibration_anchor == "bump_to_stop":
            if self.config.side not in ("left", "right"):
                raise ValueError(
                    "calibration_anchor='bump_to_stop' requires config.side='left' or 'right' "
                    "(the bump sequence and stop angles are side-specific)."
                )
            homing_offsets = run_bump_to_stop_calibration(
                self.bus,
                side=self.config.side,
                joint_limits=self.config.joint_limits,
                stop_angles_override=self.config.bump_stop_angles_deg,
                torque_thresholds_nm=self.config.bump_torque_thresholds_nm,
                velocity_thresholds_deg_s=self.config.bump_velocity_thresholds_deg_s,
                label=str(self),
            )
        else:
            self.bus.disable_torque()

            # Step 1: capture the reference pose (software homing offsets)
            input(
                "\nCalibration: capture reference (persistent software zero)\n"
                "If a motor-side zero was ever written this session, POWER-CYCLE the arm first\n"
                "(so the captured offsets reference the persistent boot frame).\n"
                "Position the arm in the following configuration:\n"
                "  - Arm hanging straight down\n"
                "  - Gripper closed\n"
                "Press ENTER when ready..."
            )

            raw_positions = self.bus.read_raw_positions()
            missing = [motor for motor in self.bus.motors if motor not in raw_positions]
            if missing:
                raise RuntimeError(
                    f"Calibration aborted: no position response from {missing}. "
                    "Check CAN wiring / motor power and retry."
                )
            homing_offsets = raw_positions

        logger.info(
            "Captured homing offsets (deg): "
            + ", ".join(f"{m}={v:.2f}" for m, v in homing_offsets.items())
        )
        for motor_name, offset in homing_offsets.items():
            if abs(offset) > 120.0:
                logger.warning(
                    f"{motor_name}: homing offset {offset:.1f} deg is close to the ±180 deg "
                    "single-turn window; boot-time readings may wrap. Consider burning the motor "
                    "zero once at this pose (openarm-can-cli set_zero), power-cycling, and "
                    "re-running this calibration so offsets end up near 0."
                )

        logger.info("Setting range: -90° to +90° for safety by default for all joints")
        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor_name],
                range_min=-90,
                range_max=90,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        # TODO(Steven, Pepijn): Slightly different from what it is happening in the leader
        with self.bus.torque_disabled():
            self.bus.configure_motors()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Get current observation from robot including position, velocity, and torque.

        Reads all motor states (pos/vel/torque) in one CAN refresh cycle
        instead of 3 separate reads.
        """
        start = time.perf_counter()

        obs_dict: dict[str, Any] = {}

        states = self.bus.sync_read_all_states()

        for motor in self.bus.motors:
            state = states.get(motor, {})
            obs_dict[f"{motor}.pos"] = state.get("position", 0.0)
            if self.config.use_velocity_and_torque:
                obs_dict[f"{motor}.vel"] = state.get("velocity", 0.0)
                obs_dict[f"{motor}.torque"] = state.get("torque", 0.0)

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        """
        Send action command to robot.

        The action magnitude may be clipped based on safety limits.

        Args:
            action: Dictionary with motor positions (e.g., "joint_1.pos", "joint_2.pos")
            custom_kp: Optional custom kp gains per motor (e.g., {"joint_1": 120.0, "joint_2": 150.0})
            custom_kd: Optional custom kd gains per motor (e.g., {"joint_1": 1.5, "joint_2": 2.0})

        Returns:
            The action actually sent (potentially clipped)
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        # Velocity feed-forward: pass the leader's joint velocity into the MIT `vel`
        # term so Kd acts as velocity tracking (smooth) rather than pure damping
        # (which causes stick-slip "buzz"). Matches the official C++ teleop, whose
        # follower MIT uses the leader velocity. Empty if the teleop doesn't emit
        # `.vel` (use_velocity_and_torque=False) -> falls back to 0.0 (legacy).
        goal_vel = {key.removesuffix(".vel"): val for key, val in action.items() if key.endswith(".vel")}

        # Apply joint limit clipping to arm
        for motor_name, position in goal_pos.items():
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f}° to {clipped_position:.2f}°")
                goal_pos[motor_name] = clipped_position

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # TODO(Steven, Pepijn): Refactor writing
        # Motor name to index mapping for gains
        motor_index = {
            "joint_1": 0,
            "joint_2": 1,
            "joint_3": 2,
            "joint_4": 3,
            "joint_5": 4,
            "joint_6": 5,
            "joint_7": 6,
            "gripper": 7,
        }

        # Use batch MIT control for arm (sends all commands, then collects responses)
        commands = {}
        for motor_name, position_degrees in goal_pos.items():
            idx = motor_index.get(motor_name, 0)
            # Use custom gains if provided, otherwise use config defaults
            if custom_kp is not None and motor_name in custom_kp:
                kp = custom_kp[motor_name]
            else:
                kp = (
                    self.config.position_kp[idx]
                    if isinstance(self.config.position_kp, list)
                    else self.config.position_kp
                )
            if custom_kd is not None and motor_name in custom_kd:
                kd = custom_kd[motor_name]
            else:
                kd = (
                    self.config.position_kd[idx]
                    if isinstance(self.config.position_kd, list)
                    else self.config.position_kd
                )
            vel_ff = goal_vel.get(motor_name)
            commands[motor_name] = (
                kp,
                kd,
                position_degrees,
                float(vel_ff) if vel_ff is not None else 0.0,
                0.0,
            )

        self.bus._mit_control_batch(commands)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        """Disconnect from robot."""

        # Disconnect CAN bus
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
