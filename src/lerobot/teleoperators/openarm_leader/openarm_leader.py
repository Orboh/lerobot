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
from typing import Any

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.damiao_alignment import (
    check_start_position,
    resolve_initial_pose,
    should_rezero_on_connect,
    soft_move_to_position,
)
from lerobot.motors.damiao.damiao_bump_calibration import run_bump_to_stop_calibration
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_openarm_leader import OpenArmLeaderConfig

logger = logging.getLogger(__name__)


class OpenArmLeader(Teleoperator):
    """
    OpenArm Leader/Teleoperator Arm with Damiao motors.

    This teleoperator uses CAN bus communication to read positions from
    Damiao motors that are manually moved (torque disabled).
    """

    config_class = OpenArmLeaderConfig
    name = "openarm_leader"

    def __init__(self, config: OpenArmLeaderConfig):
        super().__init__(config)
        if self.id is None:
            raise ValueError(
                f"{self.name} requires an explicit id "
                "(e.g. --teleop.id=leader_right). Without one the calibration path "
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

        # Gravity-compensation model (built lazily in configure() when enabled).
        self._grav_pin = None
        self._grav_model = None
        self._grav_data = None
        self._grav_qidx: list[int] = []
        self._grav_vidx: list[int] = []
        self._arm_motor_names: list[str] = []

    @property
    def action_features(self) -> dict[str, type]:
        """Features produced by this teleoperator."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """Feedback features (not implemented for OpenArms)."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the teleoperator.

        For manual control, we disable torque after connecting so the
        arm can be moved by hand.
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

        # Persistent zero: decide ONCE whether to burn a fresh motor-side zero
        # (legacy per-session behavior) or trust the calibration's software
        # homing offsets (new flow). See should_rezero_on_connect for the rule.
        rezero = should_rezero_on_connect(self.bus.calibration, self.config.rezero_on_connect)
        applies_torque = self.config.gravity_compensation or not self.config.manual_control
        if not rezero and applies_torque:
            # No re-zero and the leader is about to apply torque (gravity seed in
            # configure(), then the alignment move): refuse to move if the
            # readings look grossly out of frame (stale zero after power cycle
            # etc.). Pure manual_control never applies torque, so no gate needed.
            check_start_position(
                self.bus,
                self._side_joint_limits(),
                self.config.start_position_tolerance_deg,
                label=str(self),
            )

        self.configure()

        if rezero:
            self.bus.set_zero_position()

        # Startup alignment (official AdjustPosition port): softly move to the
        # initial pose so leader and follower start matched. Needs torque, so
        # pure manual_control (torque-off) mode skips it. The target is the
        # captured "ready" pose (initial_pose_path / initial_pose_deg) or the
        # official default. In gravity mode the arm holds the pose (soft position
        # hold + gravity feed-forward) until the teleop loop's first gravity
        # injection takes over and makes it weightless again.
        if self.config.align_on_connect:
            if applies_torque:
                goal = resolve_initial_pose(
                    initial_pose_deg=self.config.initial_pose_deg,
                    initial_pose_path=self.config.initial_pose_path,
                    joint_limits=self._side_joint_limits(),
                )
                # In gravity mode, feed forward G(q) during the ramp so a raised
                # pose is actually reached and held (soft gains alone sag under
                # gravity at a lifted pose). Evaluated at the interpolated target.
                torque_ff_fn = (
                    self._gravity_tau_from_positions if self.config.gravity_compensation else None
                )
                soft_move_to_position(
                    self.bus, goal, self.config.align_duration_s, torque_ff_fn=torque_ff_fn
                )
            else:
                logger.info("align_on_connect skipped: manual_control keeps torque disabled.")

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if teleoperator is calibrated."""
        return self.bus.is_calibrated

    def _side_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Physical joint limits for this leader's side.

        The leader config has no joint_limits field; reuse the follower's
        per-side URDF limits (function-level import to avoid a module-level
        robots<->teleoperators dependency).
        """
        from lerobot.robots.openarm_follower.config_openarm_follower import (
            LEFT_DEFAULT_JOINTS_LIMITS,
            RIGHT_DEFAULT_JOINTS_LIMITS,
        )

        return (
            RIGHT_DEFAULT_JOINTS_LIMITS
            if self.config.gravity_side == "right"
            else LEFT_DEFAULT_JOINTS_LIMITS
        )

    def soft_move_to(self, pose_deg: dict[str, float], duration_s: float = 2.0) -> bool:
        """Softly drive the leader to ``pose_deg`` on its own bus.

        Same primitive as the connect-time alignment: MIT position commands with
        the soft AdjustPosition gains, plus the gravity feed-forward in gravity
        mode so a raised pose is actually reached instead of sagging short of it.
        Exists because the generic teleop drive path goes through
        ``send_feedback``, which OpenArm does not implement — without this the
        per-episode start-pose return silently does nothing.

        Torque is deliberately left on afterwards: the teleop loop's next gravity
        injection makes the arm weightless again, whereas disabling torque here
        would drop it under its own weight.

        Returns False when the leader is not under torque (pure manual_control),
        where driving it would leave it stiff and unusable for teleoperation.
        """
        if not (self.config.gravity_compensation or not self.config.manual_control):
            logger.warning(
                "soft_move_to: leader runs in torque-off manual_control; refusing to drive it "
                "(enabling torque here would leave the arm stiff). Move it by hand instead."
            )
            return False

        goal = resolve_initial_pose(
            initial_pose_deg=pose_deg,
            joint_limits=self._side_joint_limits(),
        )
        torque_ff_fn = self._gravity_tau_from_positions if self.config.gravity_compensation else None
        soft_move_to_position(self.bus, goal, duration_s, torque_ff_fn=torque_ff_fn)
        return True

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms leader (persistent software zero).

        Two anchor modes (config.calibration_anchor), same semantics as
        OpenArmFollower.calibrate:

        "hang_down" (default):
        1. Disable torque (if not already disabled)
        2. Ask user to position the arm hanging straight down, gripper closed
        3. Read the motors' NATIVE positions at that reference and store them as
           software homing offsets in the calibration file (logical zero =
           hang-down = URDF zero). The motor-side zero (Damiao 0xFE) is
           intentionally NOT burned — see OpenArmFollower.calibrate for why.
        4. Save calibration (with the fixed ±90° default range)

        "bump_to_stop" (opt-in): sweep each ARM joint into its mechanical hard
        stop and anchor the zero there (offset = raw_at_stop - stop_angle);
        gripper never bumped, offset pinned to 0 (hand flash-zero it first with
        openarm_gripper_zero.py). The sequence side comes from
        config.gravity_side — set it explicitly. See
        lerobot.motors.damiao.damiao_bump_calibration.
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
            if self.config.gravity_side not in ("left", "right"):
                raise ValueError(
                    "calibration_anchor='bump_to_stop' requires gravity_side='left' or 'right' "
                    "(the bump sequence and stop angles are side-specific)."
                )
            homing_offsets = run_bump_to_stop_calibration(
                self.bus,
                side=self.config.gravity_side,
                joint_limits=self._side_joint_limits(),
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

        logger.info("Setting range: -90° to +90° by default for all joints")
        # TODO(Steven, Pepijn): Check if MotorCalibration is actually needed here given that we only use Degrees
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
        """
        Configure motors for the selected control mode (three branches):

          - gravity_compensation : torque ENABLED; a gravity feed-forward torque
            G(q) is injected every cycle so the arm is (near) weightless to move
            by hand.
          - manual_control       : torque DISABLED; arm moved freely by hand,
            positions read-only.
          - otherwise            : MIT/position control (configure_motors).
        """
        if self.config.gravity_compensation:
            self._init_gravity_model()
            self.bus.enable_torque()
            # Seed one injection so the arm is held the instant torque comes on
            # (no jerk: at the hanging-down zero, G(q) ~ 0).
            self._inject_gravity()
            return
        return self.bus.disable_torque() if self.config.manual_control else self.bus.configure_motors()

    def _init_gravity_model(self) -> None:
        """Build the reduced Pinocchio model used for gravity compensation.

        Faithful to the C++ KDL reference (gravity_compasation.cpp): chain root
        ``openarm_body_link0`` -> leaf ``openarm_<side>_hand``, the 7 revolute arm
        joints only (gripper fingers excluded, matching the C++ chain leaf), with
        gravity expressed in the root frame.
        """
        if self._grav_model is not None:
            return
        cfg = self.config
        if not cfg.gravity_urdf_path:
            raise ValueError(
                "gravity_compensation=True requires gravity_urdf_path (the dynamics "
                "URDF, e.g. urdf/openarm_v10_bimanual.urdf)."
            )
        if cfg.gravity_side not in ("left", "right"):
            raise ValueError(f"gravity_side must be 'left' or 'right', got {cfg.gravity_side!r}")
        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError(
                "Gravity compensation needs Pinocchio. Install it in this environment "
                "with `pip install pin`."
            ) from e

        side = cfg.gravity_side
        arm_joints = [f"openarm_{side}_joint{i}" for i in range(1, 8)]

        full = pin.buildModelFromUrdf(cfg.gravity_urdf_path)
        # Exclude the gripper fingers from the dynamics, matching the C++ chain
        # whose leaf is the (massless) hand frame rather than the fingers.
        for name in full.names:
            if name.endswith(("_finger_joint1", "_finger_joint2")):
                full.inertias[full.getJointId(name)] = pin.Inertia.Zero()
        # Reduce to just the 7 arm joints (lock everything else at the neutral pose).
        keep = {full.getJointId(n) for n in arm_joints}
        lock = [jid for jid in range(1, full.njoints) if jid not in keep]
        model = pin.buildReducedModel(full, lock, pin.neutral(full))
        model.gravity.linear = np.asarray(cfg.gravity_vector, dtype=float)

        self._grav_pin = pin
        self._grav_model = model
        self._grav_data = model.createData()
        # q / torque indices for the 7 arm joints, in URDF joint1..7 order.
        self._grav_qidx = [model.joints[model.getJointId(n)].idx_q for n in arm_joints]
        self._grav_vidx = [model.joints[model.getJointId(n)].idx_v for n in arm_joints]
        # Motor names matching joint1..7 order (motor "joint_k" <-> URDF "..._jointk").
        # Excludes the gripper, which is not part of the gravity chain.
        self._arm_motor_names = [name for name in self.bus.motors if name != "gripper"]
        if len(self._arm_motor_names) != 7:
            raise ValueError(
                f"Expected 7 arm motors for gravity compensation, found {self._arm_motor_names!r}"
            )
        logger.info(
            f"Gravity model ready ({side} arm, {model.nv} DOF, scale={cfg.gravity_scale}, "
            f"urdf={cfg.gravity_urdf_path})"
        )

    def _gravity_tau_from_positions(self, pos_deg_by_motor: dict[str, float | None]) -> dict[str, float]:
        """Gravity feed-forward torque per ARM motor for a given joint pose (deg).

        ``pos_deg_by_motor`` maps motor name -> position in degrees (extra keys
        such as ``gripper`` are ignored; the gripper is not in the gravity chain).
        Positions from lerobot are in DEGREES; the model needs RADIANS. The leader
        zero (arm hanging straight down) coincides with the URDF zero, so positions
        map directly with no offset (verified offline via FK). Returns ``{}`` when
        the gravity model is not built (gravity_compensation off).
        """
        if self._grav_model is None:
            return {}
        pin = self._grav_pin
        q = np.zeros(self._grav_model.nq)
        for k, qi in enumerate(self._grav_qidx):
            pos_deg = pos_deg_by_motor.get(self._arm_motor_names[k])
            q[qi] = np.radians(pos_deg) if pos_deg is not None else 0.0
        g = pin.computeGeneralizedGravity(self._grav_model, self._grav_data, q)
        tau = self.config.gravity_scale * np.array([g[i] for i in self._grav_vidx])
        return {motor: float(tau[k]) for k, motor in enumerate(self._arm_motor_names)}

    def _inject_gravity(self) -> dict[str, dict[str, Any]]:
        """Read joint positions, compute the gravity torque G(q), and inject it as a
        pure MIT torque feed-forward (kp=kd=0). Returns the freshly-read motor states.

        Mirrors the C++ loop: read q -> JntToGravity(q) -> MITParam{0,0,0,0,tau}.
        """
        states = self.bus.sync_read_all_states()
        pos_by_motor = {m: states.get(m, {}).get("position") for m in self._arm_motor_names}
        tau = self._gravity_tau_from_positions(pos_by_motor)

        # Arm joints: pure gravity torque. Gripper: free (torque 0) so it can still
        # be moved by hand for teleop while we read back its position.
        commands: dict[str, tuple[float, float, float, float, float]] = {}
        for motor in self._arm_motor_names:
            commands[motor] = (0.0, 0.0, 0.0, 0.0, tau.get(motor, 0.0))
        if "gripper" in self.bus.motors:
            commands["gripper"] = (0.0, 0.0, 0.0, 0.0, 0.0)

        self.bus._mit_control_batch(commands)
        return states

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get current action from the leader arm.

        This is the main method for teleoperators - it reads the current state
        of the leader arm and returns it as an action that can be sent to a follower.

        Reads all motor states (pos/vel/torque) in one CAN refresh cycle.
        """
        start = time.perf_counter()

        action_dict: dict[str, Any] = {}

        # In gravity-compensation mode the read is fused with the torque injection
        # (read q -> compute G(q) -> send MIT torque), mirroring the C++ loop.
        # Otherwise just read pos/vel/torque in one CAN cycle.
        if self.config.gravity_compensation:
            states = self._inject_gravity()
        else:
            states = self.bus.sync_read_all_states()
        for motor in self.bus.motors:
            state = states.get(motor, {})
            action_dict[f"{motor}.pos"] = state.get("position")
            if self.config.use_velocity_and_torque:
                action_dict[f"{motor}.vel"] = state.get("velocity")
                action_dict[f"{motor}.torque"] = state.get("torque")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not yet implemented for OpenArm leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect from teleoperator."""

        # Disconnect CAN bus. For any hand-movable mode (manual or gravity
        # compensation) ensure torque is disabled first, so the arm goes limp on
        # exit / Ctrl-C and is never left actively driven.
        disable = self.config.manual_control or self.config.gravity_compensation
        self.bus.disconnect(disable_torque=disable)
        logger.info(f"{self} disconnected.")
