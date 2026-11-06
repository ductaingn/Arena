"""`flowedit run`: (world, edits, crowd settings, seed) -> edited world -> the reader's scenario -> a recorded PEDS run -> tracks.

Every run is one directory under the runs root (`$ARENA_DATA_DIR/flowedit_runs` by default):

    <run id>/run.yaml        what was run: world, base, edits, settings, seed, launch command, track summary
    <run id>/launch.log      the `arena launch` output
    <run id>/humansim/       HumanSim's own recording (`humansim.record:=true`): the bag and recording.yaml
    <run id>/tracks.parquet  one row per agent per frame, with `key`, the id that pairs an agent across runs

The run id is `<world>__<scenario>`, the scenario named after the settings digest and seed, so the same inputs land in
the same directory and a finished run is skipped unless asked to run again.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arena_simulation_setup.tree.World.write import to_yaml

from . import settings as S
from .edit import EDITS, EditType, Site, apply, dump_edits, open_site, save
from .read import read
from .read import write as write_scenario

RUNS = "flowedit_runs"
RUN_FORMAT = "flowedit-run/1"
RUN_FILE = "run.yaml"
TRACKS = "tracks.parquet"


@dataclass(frozen=True)
class Edit:
    type: EditType
    target: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Prepared:
    #: the world the run launches: saved by the edit, or the base itself without one
    site: Site
    #: the generated scenario's name inside `site`'s world
    scenario: str
    settings: dict[str, Any]
    seed: int
    reader: dict[str, Any]
    notes: list[str]

    @property
    def run_id(self) -> str:
        return f"{self.site.name}__{self.scenario}"


def scenario_name(scenario: str | None, settings: Mapping[str, Any], seed: int) -> str:
    """The generated scenario's name: one per settings and seed, so runs under other settings never overwrite it."""
    return f"flowedit_{scenario or 'world'}_{S.digest(settings)[:8]}_s{seed}"


def prepare(site: Site, base: Site, edits: Sequence[Edit], settings: Mapping[str, Any], seed: int, notes: list[str] | None = None, *, dry_run: bool = False) -> Prepared:
    """Apply `edits` to `site` and save the edited world, then read the crowd into it. `base` is the unedited world the people come from.

    A dry run applies the edits and reads the scenario without writing either."""
    notes = notes if notes is not None else []
    for edit in edits:
        site = apply(site, edit.type, edit.target, dict(edit.params))
    if edits and not dry_run:
        # reopened from where it was saved, so the scenario is written into the saved world
        site = open_site(save(site), None, None, notes)
    result = read(site, settings, seed, notes, base=base)
    name = scenario_name(site.scenario, settings, seed)
    if not dry_run:
        write_scenario(result, name)
    return Prepared(site, name, dict(settings), seed, result.provenance, notes)


def launch_command(world: str, scenario: str, record_dir: Path, extra: Sequence[str] = ()) -> list[str]:
    """`arena launch` of a headless PEDS run of `scenario` in `world`, HumanSim recording into `record_dir`."""
    return [
        "arena",
        "launch",
        "sim:=dummy",
        "human:=arena",
        "headless:=true",
        f"world:={world}",
        "task.robots:=scenario",
        "task.obstacles:=scenario",
        f"task.scenario.file:={scenario}",
        "humansim.record:=true",
        f"humansim.record_dir:={record_dir}",
        *extra,
    ]


def _arena(env: dict[str, str]) -> tuple[list[str], str | None]:
    """The `arena` CLI as a command, and the directory to run it in.

    In a sourced workspace `arena` is a shell function, which a subprocess cannot call: it runs `python3 -m arena_cli`
    from $ARENA_WS_DIR with $ARENA_DIR/_meta on the path, and so does this."""
    found = shutil.which("arena", path=env.get("PATH"))
    if found:
        return [found], None
    arena_dir = env.get("ARENA_DIR")
    if not arena_dir:
        raise RuntimeError("no `arena` command and no ARENA_DIR: source the Arena workspace first")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(Path(arena_dir) / "_meta"), env.get("PYTHONPATH", "")) if p)
    return [sys.executable, "-m", "arena_cli"], env.get("ARENA_WS_DIR")


#: Nodes a run cannot go on without: the launch itself keeps running when one of them dies
VITAL = ("arena_env_node", "arena_humansim_node")


def _died(log: Path) -> str | None:
    """The first vital node `log` reports dead, else None."""
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if "process has died" in line and (node := next((n for n in VITAL if f"[{n}-" in line), None)):
            return node
    return None


def _stop(proc: subprocess.Popen, grace: float) -> None:
    """Interrupt the launch's whole process group, as Ctrl-C would, and kill what is left after `grace` seconds."""
    for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 10.0), (signal.SIGKILL, 10.0)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
        except subprocess.TimeoutExpired:
            continue


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tracks(bag: Path, duration: float | None = None) -> Any:  # noqa: ANN401 - a pandas DataFrame, pandas imported lazily
    """The agent states of `bag` keyed by `keyed`, trimmed to the first `duration` seconds of the run."""
    from arena_humansim.utils.bag_io import extract_agent_states
    from rosbags.highlevel import AnyReader

    names: dict[int, str] = {}
    with AnyReader([bag]) as reader:
        meta = [c for c in reader.connections if c.msgtype == "arena_humansim_msgs/msg/AgentMeta"]
        for connection, _, raw in reader.messages(connections=meta):
            msg = reader.deserialize(raw, connection.msgtype)
            names.update((int(i), str(n)) for i, n in zip(msg.agent_id, msg.name, strict=False) if n)
    return keyed(extract_agent_states(bag), names, duration)


def keyed(df: Any, names: Mapping[int, str], duration: float | None = None) -> Any:  # noqa: ANN401 - a pandas DataFrame
    """`df` (HumanSim's agent states) with each agent's `name` and `key`, its time from the first frame, cut at `duration`.

    `key` pairs an agent across runs: `<source_name>#<source_seq>` for a person a source released (HumanSim pairs those by
    their draws, never by agent id), the scenario name of a person the scenario places without its env prefix (the reader
    gives every edit of a world the same people), and `id<agent id>` for anything else."""
    if df.empty:
        return df
    df = df.copy()
    # HumanSim names a scenario's people inside their env (`env_1/flow_1`), and the env index is not the same every launch
    df["name"] = df["agent_id"].map({i: n.rpartition("/")[2] for i, n in names.items()}).fillna("")
    released = df["source_name"] != ""
    df["key"] = ("id" + df["agent_id"].astype(str)).where(df["name"] == "", df["name"])
    df.loc[released, "key"] = df.loc[released, "source_name"] + "#" + df.loc[released, "source_seq"].astype(str)
    df["time"] = df["time"] - df["time"].min()
    if duration is not None:
        df = df[df["time"] <= duration]
    return df.reset_index(drop=True)


def summary(df: Any) -> dict[str, Any]:  # noqa: ANN401 - a pandas DataFrame
    if df.empty:
        return {"frames": 0, "agents": 0, "seconds": 0.0}
    humans = df[df["agent_id"] > 0]
    return {
        "frames": int(df["time"].nunique()),
        "agents": int(humans["key"].nunique()),
        "released": int(humans.loc[humans["source_name"] != "", "key"].nunique()),
        "seconds": round(float(df["time"].max()), 2),
    }


def record(
    prepared: Prepared,
    out: Path,
    duration: float,
    *,
    launch_args: Sequence[str] = (),
    startup_timeout: float = 300.0,
    grace: float = 60.0,
    margin: float = 30.0,
    log=print,  # noqa: ANN001 - a print-like callable
) -> Path:
    """Launch the prepared scenario headless with HumanSim recording, stop it after `duration` seconds of frames, and write the run directory.

    The run's tracks are trimmed to the first `duration` seconds after the first frame. Raises RuntimeError when the launch
    dies or records no frames, the launch log kept for the reason."""
    run_dir = out / prepared.run_id
    humansim = run_dir / "humansim"
    if humansim.exists():
        shutil.rmtree(humansim)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_FILE).unlink(missing_ok=True)
    command = launch_command(prepared.site.name, prepared.scenario, humansim, launch_args)
    env = dict(os.environ)
    # the edited world is saved outside the source tree, the base world's generated scenario inside it: the launch sees both
    env["ARENA_WORLD_PATH"] = os.pathsep.join(p for p in (str(prepared.site.root.parent), env.get("ARENA_WORLD_PATH", "")) if p)
    arena, cwd = _arena(env)
    started = datetime.now(timezone.utc)
    log(f"{prepared.run_id}: {shlex.join(command)}")
    with (run_dir / "launch.log").open("w") as out_log:
        proc = subprocess.Popen([*arena, *command[1:]], stdout=out_log, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
        try:
            bag = humansim / "bag"
            deadline = time.monotonic() + startup_timeout
            while not bag.exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"{prepared.run_id}: the launch exited ({proc.returncode}) before HumanSim recorded, see {run_dir / 'launch.log'}")
                if died := _died(run_dir / "launch.log"):
                    raise RuntimeError(f"{prepared.run_id}: {died} died before HumanSim recorded, see {run_dir / 'launch.log'}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{prepared.run_id}: HumanSim did not start recording within {startup_timeout:.0f} s, see {run_dir / 'launch.log'}")
                time.sleep(2.0)
            log(f"{prepared.run_id}: recording, stopping in {duration:.0f} s")
            # the bag opens before the scenario loads: a margin so the trimmed tracks still span `duration`
            end = time.monotonic() + duration + margin
            while time.monotonic() < end:
                if proc.poll() is not None:
                    raise RuntimeError(f"{prepared.run_id}: the launch exited ({proc.returncode}) while recording, see {run_dir / 'launch.log'}")
                if died := _died(run_dir / "launch.log"):
                    raise RuntimeError(f"{prepared.run_id}: {died} died while recording, see {run_dir / 'launch.log'}")
                time.sleep(2.0)
        finally:
            _stop(proc, grace)
    df = tracks(humansim / "bag", duration)
    if df.empty:
        raise RuntimeError(f"{prepared.run_id}: no agent frames recorded, see {run_dir / 'launch.log'}")
    df.to_parquet(run_dir / TRACKS, index=False)
    stats = summary(df)
    doc = {
        "format": RUN_FORMAT,
        "world": prepared.site.name,
        "base": prepared.site.base,
        "scenario": prepared.site.scenario,
        "generated_scenario": prepared.scenario,
        "seed": prepared.seed,
        "settings_digest": S.digest(prepared.settings),
        "settings": prepared.settings,
        "duration": duration,
        "launch": command,
        "started": started.isoformat(timespec="seconds"),
        "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": _git_head(Path(__file__).parent),
        "edits": dump_edits(prepared.site)["edits"],
        "edit_seed": prepared.site.seed,
        "reader": prepared.reader,
        "tracks": stats,
    }
    if stats["seconds"] < duration - 1.0:
        doc["warning"] = f"tracks span {stats['seconds']} s of the {duration} s asked: the sim ran slower than real time"
        log(f"{prepared.run_id}: {doc['warning']}")
    (run_dir / RUN_FILE).write_text(to_yaml(doc))
    return run_dir


def done(run_dir: Path) -> bool:
    """Whether `run_dir` holds a finished run: run.yaml is written last."""
    return (run_dir / RUN_FILE).is_file() and (run_dir / TRACKS).is_file()


def default_out() -> Path:
    data = os.environ.get("ARENA_DATA_DIR")
    if not data:
        raise ValueError("pass --out, or set ARENA_DATA_DIR for the default runs root")
    return Path(data) / RUNS


__all__ = ["EDITS", "RUNS", "Edit", "Prepared", "default_out", "done", "keyed", "launch_command", "prepare", "record", "scenario_name", "summary", "tracks"]
