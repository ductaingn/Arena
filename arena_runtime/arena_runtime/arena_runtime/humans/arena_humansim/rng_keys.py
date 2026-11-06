"""Keys of the adapter's episode RNG streams: stable names, never the env, the env index or the spawn order.

`EpisodeRng.stream(*key)` derives each stream from (episode seed, key) by hashing, so a draw depends only on the seed and
its key. A key built from a sim path (`env_1/flow_3`) would still change with the env index, which the runtime counts up
on every spawn, and a key built from an engine agent id would change with the release order. The keys here use what is
the same in every env and every run of the same scenario: the entity's name inside its env (with its level, if any), and
for a source-released agent its source and release index, the key HumanSim itself draws that agent's streams from.
"""

from __future__ import annotations

ROOT = "humansim"


def stable_name(sim_path: str, env_prefix: str) -> str:
    """`sim_path` without its env's prefix: `env_1/flow_3` in env `env_1` is `flow_3`, `env_1/sofa/L1` is `sofa/L1`.

    A path outside `env_prefix` (or with an empty prefix) is returned as it is, leading and trailing slashes dropped."""
    path = sim_path.strip("/")
    prefix = env_prefix.strip("/")
    if prefix and (path == prefix or path.startswith(f"{prefix}/")):
        return path[len(prefix) :].lstrip("/")
    return path


def walker_key(sim_path: str, env_prefix: str) -> tuple[str, ...]:
    """Stream of a scenario pedestrian's sampled agent type (speed, radius, perception, local planner, handedness)."""
    return (ROOT, "walker", stable_name(sim_path, env_prefix))


def seats_key(sim_path: str, env_prefix: str) -> tuple[str, ...]:
    """Stream of an object's seat poses drawn from its sit slots."""
    return (ROOT, "seats", stable_name(sim_path, env_prefix))


def flow_model_key(agent_id: int, source_name: str = "", source_seq: int = 0, name: str = "") -> tuple[str | int, ...]:
    """Stream of a flow agent's human model: its source and release index, else its engine name, else its engine id.

    Only the last is unstable, for an agent the engine reports neither a source nor a name for."""
    if source_name:
        return (ROOT, "flow-model", "source", source_name, int(source_seq))
    if name:
        return (ROOT, "flow-model", "name", name)
    return (ROOT, "flow-model", "id", int(agent_id))


__all__ = ["flow_model_key", "seats_key", "stable_name", "walker_key"]
