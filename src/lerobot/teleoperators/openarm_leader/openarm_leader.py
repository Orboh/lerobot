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

        self.configure()

        if self.is_calibrated:
            self.bus.set_zero_position()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if teleoperator is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms leader.

        The calibration procedure:
        1. Disable torque (if not already disabled)
        2. Ask user to position arm in zero position (hanging with gripper closed)
        3. Set this as zero position
        4. Record range of motion for each joint
        5. Save calibration
        """
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
        self.bus.disable_torque()

        # Step 1: Set zero position
        input(
            "\nCalibration: Set Zero Position)\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )

        # Set current position as zero for all motors
        self.bus.set_zero_position()
        logger.info("Arm zero position set.")

        logger.info("Setting range: -90° to +90° by default for all joints")
        # TODO(Steven, Pepijn): Check if MotorCalibration is actually needed here given that we only use Degrees
        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
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

    def _inject_gravity(self) -> dict[str, dict[str, Any]]:
        """Read joint positions, compute the gravity torque G(q), and inject it as a
        pure MIT torque feed-forward (kp=kd=0). Returns the freshly-read motor states.

        Mirrors the C++ loop: read q -> JntToGravity(q) -> MITParam{0,0,0,0,tau}.
        Positions from lerobot are in DEGREES; the model needs RADIANS. The leader
        zero (arm hanging straight down) coincides with the URDF zero, so positions
        map directly with no offset (verified offline via FK).
        """
        pin = self._grav_pin
        cfg = self.config
        states = self.bus.sync_read_all_states()

        q = np.zeros(self._grav_model.nq)
        for k, qi in enumerate(self._grav_qidx):
            pos_deg = states.get(self._arm_motor_names[k], {}).get("position")
            q[qi] = np.radians(pos_deg) if pos_deg is not None else 0.0

        g = pin.computeGeneralizedGravity(self._grav_model, self._grav_data, q)
        tau = cfg.gravity_scale * np.array([g[i] for i in self._grav_vidx])

        # Arm joints: pure gravity torque. Gripper: free (torque 0) so it can still
        # be moved by hand for teleop while we read back its position.
        commands: dict[str, tuple[float, float, float, float, float]] = {}
        for k, motor in enumerate(self._arm_motor_names):
            commands[motor] = (0.0, 0.0, 0.0, 0.0, float(tau[k]))
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
