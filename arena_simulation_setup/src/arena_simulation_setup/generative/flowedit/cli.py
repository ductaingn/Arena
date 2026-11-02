"""`flowedit`: build the FLOWEDIT graph of a world and scenario, and read a graph back into a scenario."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from arena_simulation_setup.kg import dumps, write

from ..scenario.cli import _writable, world_root
from ..scenario.materialize import WouldOverwrite
from . import settings as S
from .build import build
from .read import read, render
from .read import write as write_scenario

FILE = "flowgraph.yaml"


def _cmd_build(args: argparse.Namespace) -> int:
    root = world_root(args.worlds, args.world)
    scenario = root / "scenarios" / args.scenario / "scenario.yaml" if args.scenario else None
    if scenario is not None and not scenario.is_file():
        raise FileNotFoundError(f"no scenario {args.scenario!r} at {scenario}")
    notes: list[str] = []
    scene = build(root, scenario, args.seed, notes)
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    if not args.write:
        print(dumps(scene), end="")
        return 0
    out = args.out or (scenario.parent if scenario is not None else root) / FILE
    write(scene, out)
    print(f"wrote {out}")
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    root = world_root(args.worlds, args.world, write=args.write)
    if args.graph is None and args.scenario is None:
        raise ValueError("name the scenario whose flowgraph.yaml to read, or pass --graph")
    graph = args.graph or root / "scenarios" / args.scenario / FILE
    if not graph.is_file():
        raise FileNotFoundError(f"no FLOWEDIT graph at {graph}, `flowedit build {args.world} {args.scenario or '<scenario>'} --write` makes one")
    name = args.name or f"flowedit_{args.scenario or graph.parent.name}"
    settings = S.load(args.settings, count=args.count, scale=args.scale, local_planner=args.local_planner)
    result = read(root, graph, settings, args.seed)
    for note in result.notes:
        print(f"note: {note}", file=sys.stderr)
    people = len(result.doc["humans"])
    print(f"{args.world}/{name}: {people} people, needs {', '.join(f'{n}->{p.target} ({p.kind}{', meet' if p.meet else ''})' for n, p in result.needs.items()) or 'none'}, entrances {result.provenance['entrances'] or 'none'}", file=sys.stderr)
    if not args.write:
        print(render(result, root, name)["scenario.yaml"], end="")
        print("(dry run - pass --write to write it)", file=sys.stderr)
        return 0
    _writable(root)
    try:
        out = write_scenario(result, root, name, force=args.force)
    except WouldOverwrite as exc:
        print(f"flowedit: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowedit", description="FLOWEDIT graphs: each object's demand on its KG node. `flowedit <verb> --help` lists a verb's options.")
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="{build,read}")
    p = verbs.add_parser("build", help="build the graph of a world and scenario", description="build the graph of a world and scenario, a dry run that prints it without --write")
    p.add_argument("world", help="the world to build from")
    p.add_argument("scenario", nargs="?", help="the scenario under the world's scenarios/ whose objects join the graph (default: world objects only)")
    p.add_argument("--seed", type=int, default=0, help="seed of the point drawn for an object posed at a zone name (default 0)")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds)")
    p.add_argument("--out", type=Path, default=None, help=f"file to write (default: {FILE} next to the scenario.yaml, or in the world without a scenario)")
    p.add_argument("--write", action="store_true", help="write the graph (default: print it)")
    p.set_defaults(fn=_cmd_build)

    p = verbs.add_parser("read", help="read a graph and crowd settings into a scenario", description="read a FLOWEDIT graph and crowd settings into scenarios/<name>/, a dry run that prints the scenario without --write")
    p.add_argument("world", help="the world the scenario is for")
    p.add_argument("scenario", nargs="?", help=f"the scenario directory whose {FILE} to read")
    p.add_argument("--graph", type=Path, default=None, help=f"graph file to read (default: scenarios/<scenario>/{FILE})")
    p.add_argument("--settings", type=Path, default=None, help="crowd settings YAML merged over the defaults (see settings.py)")
    p.add_argument("--count", type=int, default=None, help="people at the start, overriding the settings")
    p.add_argument("--scale", type=float, default=None, help="scale of populate's per-room count, overriding the settings")
    p.add_argument("--local-planner", default=None, help="HumanSim local planner of every generated type, overriding the settings")
    p.add_argument("--seed", type=int, default=0, help="seed of every draw (default 0)")
    p.add_argument("--name", default=None, help="scenario directory to write (default: flowedit_<scenario>)")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read and write (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds)")
    p.add_argument("--write", action="store_true", help="write the scenario (default: print it)")
    p.add_argument("--force", action="store_true", help="replace a scenario of that name that was not generated")
    p.set_defaults(fn=_cmd_read)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point `flowedit`."""
    args = _parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (FileNotFoundError, ValueError, yaml.YAMLError, S.SettingsError) as exc:
        print(f"flowedit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
