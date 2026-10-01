"""Gravity torque G(q) for one OpenArm side, for feed-forward during follower alignment.

Same model as the leader's gravity compensation (``OpenArmLeader._init_gravity_model``):
chain ``openarm_body_link0`` -> ``openarm_<side>_hand`` of the dynamics URDF, the 7
revolute arm joints only, gripper fingers massless. The URDF hand is the follower
hand, so the model fits the follower better than the lighter leader handle.

Why the follower needs it (2026-10-01, Itabashi): the startup alignment drives the
follower with soft gains (kp 50) and no feed-forward, so it ends ~5-6 deg below the
target under gravity (2026-08-04 J4 74.1 -> 68.6-68.9; 2026-09-22 J1 69.4 -> 63.6)
and snaps up the moment the teleop loop starts (kp 240). With a raised start pose and
many waypoints the lag is visible all the way up ("rises while sagging").
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class OpenArmGravityModel:
    def __init__(self, urdf_path: str, side: str, scale: float = 1.0,
                 gravity_vector: tuple[float, float, float] = (0.0, 0.0, -9.81)):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        import pinocchio as pin

        arm_joints = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        full = pin.buildModelFromUrdf(urdf_path)
        for name in full.names:
            if name.endswith(("_finger_joint1", "_finger_joint2")):
                full.inertias[full.getJointId(name)] = pin.Inertia.Zero()
        keep = {full.getJointId(n) for n in arm_joints}
        lock = [jid for jid in range(1, full.njoints) if jid not in keep]
        model = pin.buildReducedModel(full, lock, pin.neutral(full))
        model.gravity.linear = np.asarray(gravity_vector, dtype=float)
        self._pin, self._model, self._data = pin, model, model.createData()
        self._qidx = [model.joints[model.getJointId(n)].idx_q for n in arm_joints]
        self._vidx = [model.joints[model.getJointId(n)].idx_v for n in arm_joints]
        self.scale = scale
        logger.info(f"Follower alignment gravity model ready ({side}, scale={scale}, urdf={urdf_path})")

    def torque(self, pos_deg_by_motor: dict[str, float]) -> dict[str, float]:
        """``{joint_1..joint_7: deg}`` -> ``{joint_k: Nm}`` (gripper and unknown keys ignored)."""
        q = np.zeros(self._model.nq)
        for k, qi in enumerate(self._qidx):
            q[qi] = np.radians(pos_deg_by_motor.get(f"joint_{k + 1}", 0.0))
        g = self._pin.computeGeneralizedGravity(self._model, self._data, q)
        return {f"joint_{k + 1}": float(self.scale * g[vi]) for k, vi in enumerate(self._vidx)}
