"""A ped id that comes back under another agent type (an episode reset reuses ids) is a new person: its pose starts fresh."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from arena_people_msgs.msg import Pedestrian


def test_a_retyped_ped_starts_in_its_own_pose(rclpy_context, tmp_path: Path) -> None:
    pytest.importorskip("arena_runtime.sim.dummy_simulator")
    import tf2_ros
    import yaml
    from arena_people_msgs.msg import Pedestrians
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.acoustics.noop import NoopAcousticsSimulator
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.humans.noop import NoopHumanSimulator
    from arena_runtime.sim.dummy_simulator import DummySimulator

    walker = tmp_path / "walker.yaml"
    walker.write_text(yaml.safe_dump({"name": "walker"}))
    pusher = tmp_path / "pusher.yaml"
    pusher.write_text(yaml.safe_dump({"name": "pusher", "pose": {"walk": {"base": "walk_pushwheelchair"}}}))

    def roster(agent_type: Path) -> Pedestrians:
        ped = Pedestrian(name="person", id=1, agent_type=str(agent_type), animation_state=Pedestrian.WALKING)
        ped.pose.orientation.w = 1.0
        ped.twist.linear.x = 1.0
        return Pedestrians(pedestrians=[ped])

    class _Node(ServiceNamespace, AsyncNode):
        pass

    async def main() -> None:
        suffix = f"t_{uuid.uuid4().hex[:8]}"
        ns = Namespace(f"/test/{suffix}")
        node = _Node(f"ped_retype_{suffix}", namespace=str(ns))
        node.tf_buffer = tf2_ros.Buffer()
        try:
            realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix=""))
            simulator = DummySimulator(node=node, namespace=ns, realizer=realizer)
            human = NoopHumanSimulator(node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=NoopAcousticsSimulator(node=node, namespace=ns))
            for _ in range(5):
                human.publish_arena_peds(roster(walker))
            human.publish_arena_peds(roster(pusher))
            assert human._gait.profile_fade(1) is None, "no crossfade out of the last person's pose"
        finally:
            node.destroy_node()

    asyncio.run(main())
