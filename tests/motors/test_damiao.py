"""Minimal test script for Damiao motor with ID 3."""

import pytest

from lerobot.utils.import_utils import _can_available

if not _can_available:
    pytest.skip("python-can not available", allow_module_level=True)

import logging
import time
from collections import deque

import can
import numpy as np

from lerobot.motors import Motor
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import CAN_CMD_REFRESH, CAN_PARAM_ID, MOTOR_LIMIT_PARAMS
from lerobot.motors.motors_bus import MotorNormMode


class FakeCanBus:
    """In-memory python-can Bus stand-in simulating Damiao motors.

    Every MIT command is acked with a state frame. Refresh commands are
    answered unless the motor is configured silent (first N requests, or
    forever with -1) to exercise the re-request and give-up paths.
    """

    def __init__(self, bus: DamiaoMotorsBus) -> None:
        self._bus = bus
        self._id_to_motor = {m.id: name for name, m in bus.motors.items()}
        self.rx: deque[can.Message] = deque()
        self.positions_deg: dict[str, float] = dict.fromkeys(bus.motors, 0.0)
        self.silent_refreshes: dict[str, int] = {}  # motor -> refresh requests to ignore, -1 = all
        self.refresh_requests: dict[str, int] = dict.fromkeys(bus.motors, 0)

    def state_frame(self, motor: str, pos_deg: float) -> can.Message:
        bus = self._bus
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[bus._motor_types[motor]]
        q = bus._float_to_uint(np.radians(pos_deg), -pmax, pmax, 16)
        dq = bus._float_to_uint(0.0, -vmax, vmax, 12)
        tau = bus._float_to_uint(0.0, -tmax, tmax, 12)
        data = bytes(
            [
                0,
                (q >> 8) & 0xFF,
                q & 0xFF,
                (dq >> 4) & 0xFF,
                ((dq & 0x0F) << 4) | ((tau >> 8) & 0x0F),
                tau & 0xFF,
                30,
                30,
            ]
        )
        return can.Message(arbitration_id=bus._get_motor_recv_id(motor), data=data, is_extended_id=False)

    def send(self, msg: can.Message) -> None:
        if msg.arbitration_id == CAN_PARAM_ID and msg.data[2] == CAN_CMD_REFRESH:
            motor = self._id_to_motor[msg.data[0] | (msg.data[1] << 8)]
            self.refresh_requests[motor] += 1
            silent = self.silent_refreshes.get(motor, 0)
            if silent == -1 or self.refresh_requests[motor] <= silent:
                return
            self.rx.append(self.state_frame(motor, self.positions_deg[motor]))
        elif msg.arbitration_id in self._id_to_motor:  # MIT command ack
            motor = self._id_to_motor[msg.arbitration_id]
            self.rx.append(self.state_frame(motor, self.positions_deg[motor]))

    def recv(self, timeout: float = 0.0) -> can.Message | None:
        return self.rx.popleft() if self.rx else None


@pytest.fixture
def fake_bus():
    motors = {
        "joint_1": Motor(
            id=0x01, model="damiao", norm_mode=MotorNormMode.DEGREES, motor_type_str="dm4310", recv_id=0x11
        ),
        "joint_2": Motor(
            id=0x02, model="damiao", norm_mode=MotorNormMode.DEGREES, motor_type_str="dm4310", recv_id=0x12
        ),
        "gripper": Motor(
            id=0x08, model="damiao", norm_mode=MotorNormMode.DEGREES, motor_type_str="dm4310", recv_id=0x18
        ),
    }
    bus = DamiaoMotorsBus(port="can_test", motors=motors, can_interface="socketcan")
    fake = FakeCanBus(bus)
    bus.canbus = fake
    return bus, fake


def test_batch_refresh_updates_cache_from_responses(fake_bus):
    bus, fake = fake_bus
    fake.positions_deg["joint_1"] = 12.0
    refreshed = bus._batch_refresh(list(bus.motors))
    assert refreshed == set(bus.motors)
    assert bus._last_known_states["joint_1"]["position"] == pytest.approx(12.0, abs=0.1)
    # responsive motors are requested exactly once
    assert all(n == 1 for n in fake.refresh_requests.values())


def test_batch_refresh_rerequests_missing_motors(fake_bus):
    bus, fake = fake_bus
    fake.silent_refreshes["gripper"] = 1  # miss the first window, answer the re-request
    fake.positions_deg["gripper"] = -20.0
    refreshed = bus._batch_refresh(list(bus.motors))
    assert "gripper" in refreshed
    assert fake.refresh_requests["gripper"] == 2
    assert fake.refresh_requests["joint_1"] == 1
    assert bus._last_known_states["gripper"]["position"] == pytest.approx(-20.0, abs=0.1)
    assert bus._refresh_recovered == 1


def test_batch_refresh_gives_up_on_persistently_silent_motor(fake_bus):
    bus, fake = fake_bus
    fake.silent_refreshes["gripper"] = -1  # never answers
    for _ in range(bus.refresh_giveup_after):
        bus._batch_refresh(list(bus.motors))
    # every cycle under the threshold spends the full retry budget
    assert fake.refresh_requests["gripper"] == bus.refresh_giveup_after * (1 + bus.refresh_num_retry)
    before = fake.refresh_requests["gripper"]
    bus._batch_refresh(list(bus.motors))
    assert fake.refresh_requests["gripper"] == before + 1  # one request per cycle, no re-requests
    # when it answers again, its ack lands after the window closed (the window
    # no longer waits for it) and flows back via the drain on the NEXT cycle:
    # request->collect pipelining with zero waiting, data at most 1 cycle old
    fake.silent_refreshes["gripper"] = 0
    fake.positions_deg["gripper"] = -30.0
    bus._batch_refresh(list(bus.motors))  # ack enqueued after the window closes
    bus._batch_refresh(list(bus.motors))  # drained here
    assert bus._last_known_states["gripper"]["position"] == pytest.approx(-30.0, abs=0.1)
    assert bus._consecutive_drops["gripper"] >= bus.refresh_giveup_after  # still latched (by design)


def test_fresh_via_drain_skips_rerequest(fake_bus):
    bus, fake = fake_bus
    fake.silent_refreshes["gripper"] = -1  # refresh is never answered
    fake.rx.append(fake.state_frame("gripper", 5.0))  # but an ack from last cycle is buffered
    bus._batch_refresh(list(bus.motors))
    # the drain already made gripper's data at most one cycle old: no re-requests
    assert fake.refresh_requests["gripper"] == 1
    assert bus._refresh_fresh_via_drain == 1
    assert bus._last_known_states["gripper"]["position"] == pytest.approx(5.0, abs=0.1)


def test_latched_motor_does_not_block_the_window(fake_bus):
    bus, fake = fake_bus
    fake.silent_refreshes["gripper"] = -1
    for _ in range(bus.refresh_giveup_after):
        bus._batch_refresh(list(bus.motors))  # latch it
    t0 = time.monotonic()
    bus._batch_refresh(list(bus.motors))
    elapsed = time.monotonic() - t0
    # the window closes when the responsive motors answer; gripper is still
    # requested but its timeout is no longer paid
    assert elapsed < 0.008
    assert fake.refresh_requests["gripper"] == bus.refresh_giveup_after * (1 + bus.refresh_num_retry) + 1


def test_stale_leftover_is_drained_not_matched(fake_bus):
    bus, fake = fake_bus
    # a late response from the previous cycle is still sitting in the buffer
    fake.rx.append(fake.state_frame("gripper", 10.0))
    fake.positions_deg["gripper"] = 20.0
    refreshed = bus._batch_refresh(list(bus.motors))
    assert "gripper" in refreshed
    # the cache holds the fresh response, not the stale leftover
    assert bus._last_known_states["gripper"]["position"] == pytest.approx(20.0, abs=0.1)
    assert bus._refresh_stale_drained == 1


def test_mit_control_batch_drains_leftovers_and_acks(fake_bus):
    bus, fake = fake_bus
    fake.rx.append(fake.state_frame("joint_1", 5.0))  # leftover from a previous cycle
    fake.positions_deg["joint_1"] = 7.0
    bus._mit_control_batch({"joint_1": (10.0, 0.5, 7.0, 0.0, 0.0)})
    assert bus._last_known_states["joint_1"]["position"] == pytest.approx(7.0, abs=0.1)
    assert bus._refresh_stale_drained == 1


def test_drop_logging_is_aggregated(fake_bus, caplog):
    bus, fake = fake_bus
    fake.silent_refreshes["gripper"] = -1
    with caplog.at_level(logging.DEBUG):
        bus._batch_refresh(list(bus.motors))
        # no per-cycle drop warning
        assert not [r for r in caplog.records if "Packet drop" in r.message]
        # force the aggregation window to flush on the next cycle
        bus._refresh_last_log = time.monotonic() - 2.0
        bus._batch_refresh(list(bus.motors))
    summaries = [r for r in caplog.records if "Batch refresh" in r.message]
    assert len(summaries) == 1
    assert "gripper 2/2" in summaries[0].message


@pytest.mark.skip(reason="Requires physical Damiao motor and CAN interface")
def test_damiao_motor():
    motors = {
        "joint_3": Motor(
            id=0x03,
            model="damiao",
            norm_mode="degrees",
            motor_type_str="dm4310",
            recv_id=0x13,
        ),
    }

    bus = DamiaoMotorsBus(port="can0", motors=motors)

    try:
        print("Connecting...")
        bus.connect()
        print("✓ Connected")

        print("Enabling torque...")
        bus.enable_torque()
        print("✓ Torque enabled")

        print("Reading all states...")
        states = bus.sync_read_all_states()
        print(f"✓ States: {states}")

        print("Reading position...")
        positions = bus.sync_read("Present_Position")
        print(f"✓ Position: {positions}")

        print("Testing MIT control batch...")
        current_pos = states["joint_3"]["position"]
        commands = {"joint_3": (10.0, 0.5, current_pos, 0.0, 0.0)}
        bus._mit_control_batch(commands)
        print("✓ MIT control batch sent")

        print("Disabling torque...")
        bus.disable_torque()
        print("✓ Torque disabled")

        print("Setting zero position...")
        bus.set_zero_position()
        print("✓ Zero position set")

    finally:
        print("Disconnecting...")
        bus.disconnect(disable_torque=True)
        print("✓ Disconnected")


if __name__ == "__main__":
    test_damiao_motor()
