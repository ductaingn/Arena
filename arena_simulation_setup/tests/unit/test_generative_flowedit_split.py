from __future__ import annotations

import copy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from arena_simulation_setup.generative.flowedit import run as R
from arena_simulation_setup.generative.flowedit import settings as S
from arena_simulation_setup.generative.flowedit import split as SP
from arena_simulation_setup.generative.flowedit.cli import main

SPLITS = Path(SP.__file__).parent / "splits"


@pytest.fixture
def doc() -> dict[str, Any]:
    return {
        "format": SP.SPLIT_FORMAT,
        "sources": {"hospital_1": {"scenario": "queue", "seed": 0, "per_type": 2}, "office_1": {"scenario": "default"}},
        "duration": 5,
        "seeds": [0, 1],
        "partitions": {"train": ["hospital_1", "hospital_1__aa"], "held_out_edits": ["hospital_1__bb"], "held_out_worlds": ["office_1"]},
    }


@pytest.mark.parametrize("path", sorted(SPLITS.glob("*.yaml")), ids=lambda p: p.stem)
def test_shipped_splits_hold_out_what_they_say(path: Path) -> None:
    doc = SP.load(path)
    assert set(doc["partitions"]) == set(SP.PARTITIONS)
    assert all(doc["partitions"][p] for p in SP.PARTITIONS)


def test_a_split_keeps_its_partitions_apart(doc: dict[str, Any]) -> None:
    SP.check(doc)
    for change, message in (
        (lambda d: d["partitions"]["held_out_edits"].append("hospital_1__aa"), "both train and held_out_edits"),
        (lambda d: d["partitions"]["held_out_worlds"].append("hospital_1__cc"), r"held-out worlds \['hospital_1'\] also run in train"),
        (lambda d: d["partitions"]["held_out_edits"].append("office_1__dd"), r"held-out edits of \['office_1'\]"),
        (lambda d: d["partitions"]["train"].append("airport"), "no sources entry"),
        (lambda d: d["partitions"].update(test=[]), "unknown partitions"),
        (lambda d: d["sources"]["office_1"].update(steps_per=2), "unknown keys"),
        (lambda d: d.update(format="x"), "format"),
        (lambda d: d.update(seeds=[]), "seeds"),
    ):
        bad = copy.deepcopy(doc)
        change(bad)
        with pytest.raises(ValueError, match=message):
            SP.check(bad)


def test_every_world_runs_once_per_seed(doc: dict[str, Any]) -> None:
    runs = SP.jobs(doc)
    assert [(j.partition, j.world, j.seed) for j in runs[:2]] == [("train", "hospital_1", 0), ("train", "hospital_1", 1)]
    assert len(runs) == 8 and len(SP.jobs(doc, ["held_out_worlds"])) == 2
    base, edited = runs[0], runs[2]
    assert base.argv("queue", Path("/o"), 5.0, None)[:3] == ["run", "hospital_1", "queue"]
    assert edited.argv("queue", Path("/o"), 5.0, Path("s.yaml"))[:2] == ["run", "hospital_1__aa"] and edited.argv("queue", Path("/o"), 5.0, Path("s.yaml"))[-2:] == ["--settings", "s.yaml"]
    assert "queue" not in edited.argv("queue", Path("/o"), 5.0, None)
    assert SP.generate_argv("hospital_1", doc["sources"]["hospital_1"]) == ["generate", "hospital_1", "--scenario", "queue", "--seed", "0", "--per-type", "2", "--steps", "1", "--write"]


def test_a_sweep_runs_every_job_and_counts_how_each_ended(doc: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a stand-in `flowedit`: office_1 fails, hospital_1__bb is already recorded, every launch prints its ROS domain
    script = "import os, sys; a = sys.argv[1:]; print('domain', os.environ['ROS_DOMAIN_ID']); w = a[1]; print('already recorded' if w == 'hospital_1__bb' else ''); sys.exit(3 if w == 'office_1' else 0)"

    def fake(args: Sequence[str]) -> list[str]:
        return [sys.executable, "-c", script, *args]

    monkeypatch.setattr(SP, "_flowedit", fake)
    # a finished run (run.yaml and tracks) is skipped without a launch
    finished = tmp_path / f"hospital_1__{R.scenario_name('queue', S.load(), 1)}"
    finished.mkdir(parents=True)
    (finished / R.RUN_FILE).write_text("format: flowedit-run/1\n")
    (finished / R.TRACKS).write_text("")
    counts = SP.sweep(doc, tmp_path, jobs_at_once=2, domain_base=170, regenerate=False, stagger=0.0, log=lambda _: None)
    assert counts == {"done": 3, "skipped": 3, "failed": 2}
    logs = sorted((tmp_path / "_sweep").iterdir())
    assert len(logs) == 7 and {line for f in logs for line in f.read_text().splitlines() if line.startswith("domain")} == {"domain 170", "domain 171"}


def test_cli_lists_a_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sweep", str(SPLITS / "peds_v1.yaml"), "--partition", "held_out_worlds", "--dry-run"]) == 0
    rows = [line.split("\t") for line in capsys.readouterr().out.splitlines()]
    assert rows and {r[0] for r in rows} == {"held_out_worlds"} and rows[0][1] == "office_1"
