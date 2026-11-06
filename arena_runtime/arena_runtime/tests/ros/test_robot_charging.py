"""A charger charges the first robot parked in its dock with its power, 0 W for any other robot."""

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

    from arena_runtime.env.robot_manager.charging import PARK_TIME, Chargers
    from arena_runtime.env.robot_manager.robot_manager import _LATCHED, RobotManager
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

DOCK = [[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]]
# every object faces -x, so each dock spans x 2.5..3.5: charger_1 y 1.5..2.5, the locker (no charger) y 3.5..4.5, charger_2 y 5.5..6.5
CHARGERS = {"charger_1": 1620.0, "charger_2": 800.0}
TICK = 1.0 / 30.0
TOPIC = "/test_robot_charging/jackal_0/power_publisher/charge_power"


def _zones() -> Zones:
    def obstacle(name: str, y: float, power: float | None) -> Obstacle:
        return Obstacle(name=name, model=name, pose=Pose(position=Position(4.0, y), orientation=Orientation.from_yaw(math.pi)), dock=DOCK, charge_power_w=power)

    return zone_polygons(Level(id="0", objects=[obstacle("charger_1", 2.0, 1620.0), obstacle("locker_1", 4.0, None), obstacle("charger_2", 6.0, 800.0)]))


def _at(t: float, **robots: tuple[float, float]) -> Sample:
    return Sample(t=t, robots={name: (x, y, 0.0) for name, (x, y) in robots.items()})


def test_a_robot_charges_once_parked_in_a_chargers_dock_and_not_while_driving_through() -> None:
    chargers, zones = Chargers(CHARGERS), _zones()
    for step in range(121):
        assert chargers.step(_at(step * TICK, jackal_0=(2.4 + step * 0.01, 2.0)), zones) == {"jackal_0": 0.0}
    stop = 121 * TICK
    assert chargers.step(_at(stop, jackal_0=(3.0, 2.0)), zones) == {"jackal_0": 0.0}
    assert chargers.step(_at(stop + PARK_TIME / 2, jackal_0=(3.0, 2.0)), zones) == {"jackal_0": 0.0}
    assert chargers.step(_at(stop + PARK_TIME, jackal_0=(3.0, 2.0)), zones) == {"jackal_0": 1620.0}
    assert chargers.step(_at(stop + PARK_TIME + 1.0, jackal_0=(3.0, 2.04)), zones) == {"jackal_0": 1620.0}


def test_a_robot_parked_in_a_docked_object_without_charge_power_does_not_charge() -> None:
    chargers, zones = Chargers(CHARGERS), _zones()
    chargers.step(_at(0.0, jackal_0=(3.0, 4.0)), zones)
    assert chargers.step(_at(PARK_TIME, jackal_0=(3.0, 4.0)), zones) == {"jackal_0": 0.0}


def test_a_charger_charges_the_first_robot_parked_in_its_dock_until_it_leaves() -> None:
    chargers, zones = Chargers(CHARGERS), _zones()
    chargers.step(_at(0.0, jackal_1=(1.0, 2.0), jackal_0=(3.0, 2.0), jackal_2=(3.0, 6.0)), zones)
    assert chargers.step(_at(PARK_TIME / 2, jackal_1=(3.3, 2.3), jackal_0=(3.0, 2.0), jackal_2=(3.0, 6.0)), zones) == {"jackal_1": 0.0, "jackal_0": 0.0, "jackal_2": 0.0}
    assert chargers.step(_at(PARK_TIME, jackal_1=(3.3, 2.3), jackal_0=(3.0, 2.0), jackal_2=(3.0, 6.0)), zones) == {"jackal_1": 0.0, "jackal_0": 1620.0, "jackal_2": 800.0}
    assert chargers.step(_at(2 * PARK_TIME, jackal_1=(3.3, 2.3), jackal_0=(3.0, 2.0), jackal_2=(3.0, 6.0)), zones) == {"jackal_1": 0.0, "jackal_0": 1620.0, "jackal_2": 800.0}
    assert chargers.step(_at(3 * PARK_TIME, jackal_1=(3.3, 2.3), jackal_0=(1.0, 2.0), jackal_2=(3.0, 6.0)), zones) == {"jackal_1": 1620.0, "jackal_0": 0.0, "jackal_2": 800.0}


def test_the_charge_power_goes_out_on_change() -> None:
    node = rclpy.node.Node("test_robot_charging")
    received: list[float] = []
    node.create_subscription(std_msgs.msg.Float64, TOPIC, lambda msg: received.append(msg.data), _LATCHED)
    manager = RobotManager.__new__(RobotManager)
    manager._charge_power_w = 0.0
    manager._charge_pub = node.create_publisher(std_msgs.msg.Float64, TOPIC, _LATCHED)

    def spin_until(count: int) -> None:
        deadline = time.monotonic() + 5.0
        while len(received) < count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

    try:
        manager.publish_charge_power(0.0)
        manager.publish_charge_power(1620.0)
        spin_until(1)
        manager.publish_charge_power(1620.0)
        manager.publish_charge_power(0.0)
        spin_until(2)
        assert received == [1620.0, 0.0]
    finally:
        node.destroy_node()
