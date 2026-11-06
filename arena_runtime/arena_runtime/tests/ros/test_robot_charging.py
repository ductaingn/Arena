"""A robot in a charger's dock sends the charger's power to its power publisher, 0 W once it leaves."""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import pytest

try:
    import rclpy
    import rclpy.node
    import std_msgs.msg
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.shared.judge import Sample, Zones, zone_polygons
    from arena_simulation_setup.tree.World import Level
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

    from arena_runtime.env.robot_manager.robot_manager import _LATCHED, RobotManager, charge_power
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

ROBOT = "jackal_0"
DOCK = [[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]]
# every object faces -x, so each dock spans x 2.5..3.5: charger_1 y 1.5..2.5, the locker (no charger) y 3.5..4.5, charger_2 y 5.5..6.5
CHARGERS = {"charger_1": 1620.0, "charger_2": 800.0}
TOPIC = "/test_robot_charging/jackal_0/power_publisher/charge_power"


def _zones() -> Zones:
    poses = {"charger_1": Position(4.0, 2.0), "charger_2": Position(4.0, 6.0)}
    chargers = [Obstacle(name=name, model="charger", pose=Pose(position=poses[name], orientation=Orientation.from_yaw(math.pi)), dock=DOCK, charge_power_w=power) for name, power in CHARGERS.items()]
    locker = Obstacle(name="locker_1", model="locker", pose=Pose(position=Position(4.0, 4.0), orientation=Orientation.from_yaw(math.pi)), dock=DOCK)
    return zone_polygons(Level(id="0", objects=[*chargers, locker]))


def _at(x: float, y: float) -> Sample:
    return Sample(t=0.0, robots={ROBOT: (x, y, 0.0)})


def test_a_robot_charges_with_the_power_of_the_charger_whose_dock_holds_it() -> None:
    zones = _zones()
    assert charge_power(ROBOT, _at(3.0, 2.0), zones, CHARGERS) == 1620.0
    assert charge_power(ROBOT, _at(3.0, 6.0), zones, CHARGERS) == 800.0
    assert charge_power(ROBOT, _at(3.0, 4.0), zones, CHARGERS) == 0.0
    assert charge_power(ROBOT, _at(1.0, 2.0), zones, CHARGERS) == 0.0
    assert charge_power(ROBOT, Sample(t=0.0), zones, CHARGERS) == 0.0


def test_the_charge_power_goes_out_on_change_and_drops_to_zero_off_the_dock() -> None:
    node = rclpy.node.Node("test_robot_charging")
    received: list[float] = []
    node.create_subscription(std_msgs.msg.Float64, TOPIC, lambda msg: received.append(msg.data), _LATCHED)
    manager = RobotManager.__new__(RobotManager)
    manager._robot = SimpleNamespace(name=ROBOT)
    manager._charge_power_w = 0.0
    manager._charge_pub = node.create_publisher(std_msgs.msg.Float64, TOPIC, _LATCHED)
    zones = _zones()

    def spin_until(count: int) -> None:
        deadline = time.monotonic() + 5.0
        while len(received) < count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

    try:
        manager.publish_charge_power(_at(3.0, 2.0), zones, CHARGERS)
        spin_until(1)
        manager.publish_charge_power(_at(3.1, 2.1), zones, CHARGERS)
        manager.publish_charge_power(_at(3.0, 4.0), zones, CHARGERS)
        spin_until(2)
        assert received == [1620.0, 0.0]
    finally:
        node.destroy_node()
