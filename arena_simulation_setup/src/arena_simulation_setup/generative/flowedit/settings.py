"""Crowd settings: the global knobs a FLOWEDIT scenario is read with, the same for every site."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

#: Every key a settings file may set, with its default. A need's entry is merged over `needs["*"]`.
DEFAULTS: dict[str, Any] = {
    # people at the start: an absolute count, else populate's count for the rooms (2 per room of the target role) times `scale`
    "count": None,
    "scale": 1.0,
    "needs": {
        "*": {
            "initial": {"mean": 60.0, "std": 20.0, "clip_low": 20.0, "clip_high": 100.0},
            "decay_rate": {"mean": 0.5, "std": 0.15, "clip_low": 0.1, "clip_high": 2.0},
            # a person seeks the need once it falls below this
            "below": 30.0,
        },
    },
    # objects that satisfy no need get one named `visit_<service type>`, so every afforded object is used. Off by default:
    # a crowd seeking a capacity-1 chair or bed waits where it stands, and those waiting at the fountain block it
    "synthesize_needs": False,
    # how long a person tries to reach and use an object before giving up (nav + queue + use)
    "patience": {"mean": 120.0, "std": 30.0, "clip_low": 30.0, "clip_high": 300.0},
    # dwell for an object the graph gives no `dwells`
    "dwell": {"mean": 10.0, "std": 3.0, "clip_low": 3.0, "clip_high": 30.0},
    # idle walking between rooms of the person's route, standing this long at each stop
    "idle": {"stand": {"mean": 8.0, "std": 3.0, "clip_low": 2.0, "clip_high": 20.0}},
    # HumanSim local planner of every generated type (null: the launch default) and its parameter overrides
    "local_planner": None,
    "local_planner_params": {},
    # people entering through every exterior door and leaving through it again; rate 0 writes no regions
    "entrances": {"rate": 0.05, "max_concurrent": 4, "max_total": 15, "speed": {"min": 1.0, "max": 1.5}, "stops": 2},
    # robot start and goal (zone names or [x, y(, yaw)]), null: copied from a hand-written scenario of the world
    "robot": None,
}


class SettingsError(ValueError):
    pass


#: Settings whose keys are free (need names, planner parameters, robot fields) rather than checked against the defaults.
FREE = ("needs", "local_planner_params", "robot")
NEED_KEYS = ("initial", "decay_rate", "below")


def _merge(base: Mapping[str, Any], over: Mapping[str, Any], path: str = "", free: bool = False) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in over.items():
        where = f"{path}{key}"
        if key not in out and not free:
            raise SettingsError(f"unknown crowd setting {where!r}, known: {sorted(out)}")
        if isinstance(out.get(key), Mapping) and isinstance(value, Mapping):
            out[key] = _merge(out[key], value, f"{where}.", free or (not path and key in FREE))
        else:
            out[key] = copy.deepcopy(value)
    return out


def load(source: str | Path | Mapping[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    """The defaults with `source` (a YAML file or mapping) and then `overrides` merged over them, unknown keys refused."""
    raw: Mapping[str, Any] = {}
    if isinstance(source, Mapping):
        raw = source
    elif source is not None:
        raw = yaml.safe_load(Path(source).read_text()) or {}
        if not isinstance(raw, Mapping):
            raise SettingsError(f"{source}: crowd settings are a mapping")
    out = _merge(DEFAULTS, raw)
    out = _merge(out, {k: v for k, v in overrides.items() if v is not None})
    if out["count"] is not None and int(out["count"]) < 0:
        raise SettingsError("count is a number of people, at least 0")
    if float(out["scale"]) < 0:
        raise SettingsError("scale is at least 0")
    return out


def need(settings: Mapping[str, Any], name: str) -> dict[str, Any]:
    """The settings of need `name`: each key of its own entry over the same key of `needs["*"]`."""
    entry = settings["needs"].get(name) or {}
    if unknown := sorted(set(entry) - set(NEED_KEYS)):
        raise SettingsError(f"needs.{name}: unknown keys {unknown}, known: {list(NEED_KEYS)}")
    # a need's own distribution replaces the default one whole: a raised mean must not keep the default's clips
    return {**copy.deepcopy(settings["needs"]["*"]), **copy.deepcopy(dict(entry))}


def digest(settings: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
