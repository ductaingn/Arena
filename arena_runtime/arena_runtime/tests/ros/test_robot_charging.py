"""A charger charges the first robot parked in its dock with its power, and each robot starts an episode with a charge drawn per seed and name."""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import pytest

try:
    import rclpy
    import rclpy.node
    import std_msgs.msg
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_rclpy_mixins.shared import Namespace
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.shared.judge import Sample, Zones, zone_polygons
    from arena_simulation_setup.tree.World import Level
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

    from arena_runtime.env.constants.rng import EpisodeRng
    from arena_runtime.env.constants.runtime import Configuration
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


def _manager_in_env(env: int, name: str) -> RobotManager:
    manager = RobotManager.__new__(RobotManager)
    manager._namespace = Namespace(f"/arena/env_{env}/arena_env_node")
    manager._robot = SimpleNamespace(name=name)
    return manager


def _rng(seed: int) -> EpisodeRng:
    rng = EpisodeRng()
    rng.reseed(seed)
    return rng


def test_the_initial_charge_is_drawn_per_seed_and_robot_name_whatever_the_env_index() -> None:
    names = ("jackal", "jackal_1", "jackal_2", "jackal_3")
    socs = {(seed, name): _manager_in_env(0, name).initial_soc(_rng(seed), 40.0, 100.0) for seed in range(1, 11) for name in names}
    assert all(40.0 <= soc <= 100.0 for soc in socs.values())
    assert len(set(socs.values())) == len(socs)
    for env in (1, 7):
        assert _manager_in_env(env, "jackal").namespace == f"/arena/env_{env}/arena_env_node/jackal"
        assert _manager_in_env(env, "jackal").initial_soc(_rng(3), 40.0, 100.0) == socs[(3, "jackal")]


def test_the_initial_charge_ignores_the_episodes_other_draws() -> None:
    rng = _rng(3)
    rng.stream("robots", "target", "jackal").uniform()
    rng.stream("robot-adapter", "jackal").uniform()
    assert _manager_in_env(0, "jackal").initial_soc(rng, 40.0, 100.0) == _manager_in_env(0, "jackal").initial_soc(_rng(3), 40.0, 100.0)


def _soc_range(**soc: float) -> tuple[float, float]:
    node = ROSParamServer("test_battery_soc", parameter_overrides=[rclpy.Parameter("world", value="map_empty"), *(rclpy.Parameter(f"task.episode.battery.soc.{key}", value=value) for key, value in soc.items())])
    try:
        robot = Configuration(node).Robot
        return robot.BATTERY_SOC_MIN.value, robot.BATTERY_SOC_MAX.value
    finally:
        node.destroy_node()


def test_the_start_charge_range_defaults_to_40_to_100_percent_and_takes_launch_values() -> None:
    assert _soc_range() == (40.0, 100.0)
    assert _soc_range(min=0.0, max=55.5) == (0.0, 55.5)


@pytest.mark.parametrize(("key", "value"), [("min", -1.0), ("max", 150.0)])
def test_a_start_charge_bound_outside_0_to_100_percent_is_refused(key: str, value: float) -> None:
    with pytest.raises(RuntimeError, match=f"task.episode.battery.soc.{key} failed"):
        _soc_range(**{key: value})


def test_a_start_charge_range_with_its_min_above_its_max_is_refused() -> None:
    with pytest.raises(ValueError, match=r"task.episode.battery.soc.min \(80.0\) is above task.episode.battery.soc.max \(40.0\)"):
        _manager_in_env(0, "jackal").initial_soc(_rng(3), 80.0, 40.0)
