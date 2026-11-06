"""Flatsim pedestrian discs take each pedestrian's own radius and keep it through updates that carry none."""

from __future__ import annotations

import asyncio
import typing

import pytest


async def _until(pred: typing.Callable[[], bool], what: str, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.02)


def _peds(*peds: tuple[str, float, float]) -> typing.Any:
    from arena_people_msgs.msg import Pedestrian, Pedestrians

    msg = Pedestrians()
    for name, x, radius in peds:
        ped = Pedestrian(name=name, radius=radius)
        ped.pose.position.x = x
        ped.pose.orientation.w = 1.0
        msg.pedestrians.append(ped)
    return msg


def test_a_disc_takes_the_pedestrians_radius_ped_radius_without_one_and_keeps_its_size(on_flatsim):
    from arena_runtime.humans import PED_RADIUS

    async def body(sim, world):
        wheelchair, unknown = sim._name("wheelchair"), sim._name("unknown")

        def disc(name: str) -> tuple[float, float] | None:
            entity = world._entities.get(name)
            return None if entity is None else (entity.radius, entity.pose[0])

        assert list(await sim.pedestrian_update(_peds(("wheelchair", 1.0, 0.325), ("unknown", 3.0, 0.0)))) == [True, True]
        await _until(lambda: disc(wheelchair) is not None and disc(unknown) is not None, "both discs")
        assert disc(wheelchair) == pytest.approx((0.325, 1.0)) and disc(unknown) == pytest.approx((PED_RADIUS, 3.0))
        # a teleport or a possessed ped comes without a radius
        await sim.pedestrian_update(_peds(("wheelchair", 2.0, 0.0)))
        await _until(lambda: disc(wheelchair) == pytest.approx((0.325, 2.0)), "the moved disc at its own size")
        await sim.pedestrian_update(_peds(("unknown", 3.0, 0.18)))
        await _until(lambda: disc(unknown) == pytest.approx((0.18, 3.0)), "the disc resized once its radius is known")

    on_flatsim(body)
