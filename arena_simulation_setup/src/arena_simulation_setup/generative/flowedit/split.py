"""Splits of edited worlds into train, held-out edits and held-out worlds, and the sweep that records every run of one.

A split file names worlds only. Every edited world in it is made again, under the same name, by `flowedit generate` with
its base's `sources` entry, so the file is all a sweep needs:

    format: flowedit-split/1
    sources:                 # base world: the generate call its edited worlds come from
      hospital_1: {scenario: queue, seed: 0, per_type: 3, steps: 1}
    duration: 300            # seconds recorded per run
    seeds: [0]               # crowd seeds, every world runs once per seed
    partitions:
      train: [hospital_1, hospital_1__9f4189dc, ...]
      held_out_edits: [...]  # other edits of the train worlds
      held_out_worlds: [...] # worlds, and their edits, with no run in train
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SPLIT_FORMAT = "flowedit-split/1"
PARTITIONS = ("train", "held_out_edits", "held_out_worlds")
SOURCE_KEYS = ("scenario", "seed", "per_type", "steps")


def base_of(world: str) -> str:
    """The base world of an edited world's name (`<base>__<edit id>`), or the name itself."""
    return world.partition("__")[0]


def check(doc: Mapping[str, Any]) -> None:
    """Raise ValueError unless `doc` is a split whose partitions are disjoint and whose held-out worlds never run in train."""
    if doc.get("format") != SPLIT_FORMAT:
        raise ValueError(f"a split has format {SPLIT_FORMAT}, got {doc.get('format')!r}")
    parts = doc.get("partitions") or {}
    if unknown := sorted(set(parts) - set(PARTITIONS)):
        raise ValueError(f"unknown partitions {unknown}, known: {list(PARTITIONS)}")
    sources = doc.get("sources") or {}
    for base, source in sources.items():
        if unknown := sorted(set(source) - set(SOURCE_KEYS)):
            raise ValueError(f"sources.{base}: unknown keys {unknown}, known: {list(SOURCE_KEYS)}")
    seen: dict[str, str] = {}
    for part, worlds in parts.items():
        for world in worlds:
            if world in seen:
                raise ValueError(f"{world} is in both {seen[world]} and {part}")
            seen[world] = part
            if base_of(world) not in sources:
                raise ValueError(f"{part}: {world} has no sources entry for its base {base_of(world)}")
    train = {base_of(w) for w in parts.get("train", [])}
    if leaked := sorted({base_of(w) for w in parts.get("held_out_worlds", [])} & train):
        raise ValueError(f"held-out worlds {leaked} also run in train")
    if stray := sorted({base_of(w) for w in parts.get("held_out_edits", [])} - train):
        raise ValueError(f"held-out edits of {stray}, worlds that never run in train")
    if not isinstance(doc.get("seeds", [0]), list) or not doc.get("seeds", [0]):
        raise ValueError("seeds is a non-empty list of crowd seeds")


def load(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text()) or {}
    check(doc)
    return doc


@dataclass(frozen=True)
class Job:
    partition: str
    world: str
    seed: int

    def argv(self, scenario: str | None, out: Path, duration: float, settings: Path | None) -> list[str]:
        """`flowedit run` of this job: an edited world names its own scenario, a base world the source's."""
        args = ["run", self.world, *([scenario] if scenario and "__" not in self.world else []), "--seed", str(self.seed), "--duration", str(duration), "--out", str(out)]
        return [*args, *(["--settings", str(settings)] if settings is not None else [])]


def jobs(doc: Mapping[str, Any], partitions: Iterable[str] = PARTITIONS) -> list[Job]:
    """Every run of the chosen partitions: each world once per seed, in file order."""
    return [Job(part, world, int(seed)) for part in partitions for world in (doc["partitions"].get(part) or []) for seed in doc.get("seeds", [0])]


def generate_argv(base: str, source: Mapping[str, Any]) -> list[str]:
    """`flowedit generate` that saves the edited worlds of `base` again under the names the split gives them."""
    return [
        "generate",
        base,
        *(["--scenario", str(source["scenario"])] if source.get("scenario") else []),
        "--seed",
        str(source.get("seed", 0)),
        "--per-type",
        str(source.get("per_type", 1)),
        "--steps",
        str(source.get("steps", 1)),
        "--write",
    ]


def _flowedit(args: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", "arena_simulation_setup.generative.flowedit.cli", *args]


def sweep(
    doc: Mapping[str, Any],
    out: Path,
    *,
    partitions: Sequence[str] = PARTITIONS,
    settings: Path | None = None,
    jobs_at_once: int = 1,
    domain_base: int = 150,
    regenerate: bool = True,
    stagger: float = 30.0,
    log=print,  # noqa: ANN001 - a print-like callable
) -> dict[str, int]:
    """Record every run of the chosen partitions under `out`, `jobs_at_once` launches at a time, each on its own ROS domain.

    The edited worlds are generated again first, so the names in the split resolve. A finished run is skipped. Returns the
    count of runs that finished, were skipped and failed; a failed run's reason is in its launch.log. Launches start at
    least `stagger` seconds apart: runtimes coming up at the same moment on a loaded host have stalled before registering."""
    if regenerate:
        for base, source in doc["sources"].items():
            log(f"generating the edited worlds of {base}")
            subprocess.run(_flowedit(generate_argv(base, source)), check=True, stdout=subprocess.DEVNULL)
    from . import settings as S
    from .run import done, scenario_name

    crowd = S.load(settings)
    duration = float(doc.get("duration", 300.0))
    counts = {"done": 0, "skipped": 0, "failed": 0}
    queue = []
    for job in jobs(doc, partitions):
        # a finished run is skipped here, without a launch slot or a stagger
        if done(out / f"{job.world}__{scenario_name(doc['sources'][base_of(job.world)].get('scenario'), crowd, job.seed)}"):
            counts["skipped"] += 1
            log(f"[{job.partition}] {job.world} seed {job.seed}: already recorded")
        else:
            queue.append(job)
    running: dict[int, tuple[Job, subprocess.Popen]] = {}
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "_sweep"
    logs.mkdir(exist_ok=True)
    last_start = -float("inf")
    while queue or running:
        for slot in [s for s in range(jobs_at_once) if s not in running]:
            if not queue or time.monotonic() - last_start < stagger:
                break
            last_start = time.monotonic()
            job = queue.pop(0)
            env = {**os.environ, "ROS_DOMAIN_ID": str(domain_base + slot)}
            argv = job.argv(doc["sources"][base_of(job.world)].get("scenario"), out, duration, settings)
            with (logs / f"{job.world}__s{job.seed}.log").open("w") as fh:
                running[slot] = (job, subprocess.Popen(_flowedit(argv), stdout=fh, stderr=subprocess.STDOUT, env=env))
            log(f"[{job.partition}] {job.world} seed {job.seed}: started on domain {domain_base + slot}")
        time.sleep(2.0)
        for slot, (job, proc) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[slot]
            text = (logs / f"{job.world}__s{job.seed}.log").read_text()
            state = "failed" if proc.returncode else "skipped" if "already recorded" in text else "done"
            counts[state] += 1
            log(f"[{job.partition}] {job.world} seed {job.seed}: {state}" + (f" ({text.strip().splitlines()[-1] if text.strip() else proc.returncode})" if state == "failed" else ""))
    return counts


__all__ = ["PARTITIONS", "SPLIT_FORMAT", "Job", "base_of", "check", "generate_argv", "jobs", "load", "sweep"]
