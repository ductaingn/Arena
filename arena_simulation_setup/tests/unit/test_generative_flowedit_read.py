from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from arena_simulation_setup.generative.flowedit import settings as S
from arena_simulation_setup.generative.flowedit.build import build
from arena_simulation_setup.generative.flowedit.cli import main
from arena_simulation_setup.generative.flowedit.interactions import by_rule
from arena_simulation_setup.generative.flowedit.read import Result, read, render, write
from arena_simulation_setup.generative.scenario.materialize import WouldOverwrite
from arena_simulation_setup.kg import Scene, Value, load
from arena_simulation_setup.kg import write as write_kg

pytest.importorskip("numpy")
pytest.importorskip("arena_humansim")

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"
GRAPH = HOSPITAL / "scenarios" / "queue" / "flowgraph.yaml"
SHOWCASE = HOSPITAL / "scenarios" / "showcase" / "scenario.yaml"


def _types(result: Result) -> dict[str, dict[str, Any]]:
    return {ref: doc for ref, doc in result.types.items() if "/flow__" in ref}


def _object(result: Result, name: str) -> dict[str, Any]:
    return next(o for o in result.doc["objects"] if o["name"] == name)


def _edited(tmp_path: Path, subject: str, verb: str, params: dict[str, Any] | None = None, obj: Value | None = None) -> Path:
    """The hospital graph with the `verb` fact of `subject` given new `params` (merged) or a new object, saved under `tmp_path`."""
    scene = load(GRAPH)
    (fact,) = [f for f in scene.facts if f.subject == subject and f.verb == verb]
    scene.remove_fact(fact.subject, fact.verb, fact.object, fact.params)
    scene.add_fact(subject, verb, fact.object if obj is None else obj, {**fact.params, **(params or {})})
    path = tmp_path / "flowgraph.yaml"
    write_kg(scene, path)
    return path


@pytest.fixture(scope="module")
def hospital() -> Result:
    return read(HOSPITAL, GRAPH)


@pytest.fixture(scope="module")
def every_object() -> Result:
    return read(HOSPITAL, GRAPH, S.load({"synthesize_needs": True}))


def test_hospital_graph_queues_at_the_fountain(hospital: Result) -> None:
    fountain = _object(hospital, "fountain")
    assert fountain["pose"] == [11.0, 10.0, 1.57]
    assert fountain["humansim"]["capacity"] == 1 and fountain["humansim"]["formation"]["type"] == "line"
    thirst = hospital.needs["thirst"]
    assert (thirst.target, thirst.kind, thirst.meet, thirst.dwell["mean"]) == ("fountain", "USE", False, 5.0)
    people = _types(hospital)
    assert people and len(people) == len(hospital.doc["humans"])
    for doc in people.values():
        assert doc["mode"] == "behavior_tree" and doc["needs"]["thirst"]
        assert {"when": {"thirst": {"below": 30.0}}, "goto": "seek_thirst"} in doc["sequences"]["idle"]["transitions"]
        assert doc["sequences"]["seek_thirst"]["steps"]["use"]["target"] == "fountain"
    assert all(h["name"].startswith("flow_") and "colleague" not in h["agent"]["agent_type"] for h in hospital.doc["humans"])


def test_only_stated_needs_are_sought_by_default(hospital: Result) -> None:
    assert list(hospital.needs) == ["thirst"]
    assert {o["name"] for o in hospital.doc["objects"]} == {"exam_chair_1", "fountain", "gathering_area", "patient_bed_1", "reception_desk"}


def test_hospital_reads_every_interaction_kind(every_object: Result) -> None:
    kinds = {n: (p.target, p.kind, p.meet) for n, p in every_object.needs.items()}
    assert kinds == {
        "thirst": ("fountain", "USE", False),
        "visit_chair": ("exam_chair_1", "SIT_ON", False),
        "visit_desk": ("reception_desk", "QUEUE_USE", False),
        "visit_gathering_area": ("gathering_area", "GROUP_CONVERSATION", True),
        "visit_patient_bed": ("patient_bed_1", "LIE_ON", False),
    }
    seek = next(iter(_types(every_object).values()))["sequences"]["seek_visit_gathering_area"]["steps"]
    assert seek["go"] == {"kind": "go_to", "target": "gathering_area", "patience": S.DEFAULTS["patience"]}
    assert seek["use"]["interaction"] == "GROUP_CONVERSATION" and seek["use"]["duration"]["mean"] == 45.0


def test_patience_outlasts_the_longest_use(every_object: Result) -> None:
    for doc in _types(every_object).values():
        for name, sequence in doc["sequences"].items():
            for step in sequence["steps"].values():
                if "duration" in step and "patience" in step:
                    assert step["patience"]["clip_low"] > step["duration"].get("clip_high", step["duration"]["mean"]), (name, step)


def test_moving_the_fountain_in_the_graph_moves_the_queue(tmp_path: Path, hospital: Result) -> None:
    moved = read(HOSPITAL, _edited(tmp_path, "obj:fountain", "at", {"x": 11.2, "y": 24.0}))
    assert _object(moved, "fountain")["pose"] == [11.2, 24.0, 1.57]
    assert moved.needs["thirst"].target == "fountain"
    assert [o for o in moved.doc["objects"] if o["name"] != "fountain"] == [o for o in hospital.doc["objects"] if o["name"] != "fountain"]


def test_raising_dwell_in_the_graph_lengthens_each_use(tmp_path: Path) -> None:
    raised = read(HOSPITAL, _edited(tmp_path, "obj:fountain", "dwells", {"clip_high": 25.0}, obj=20.0))
    assert raised.needs["thirst"].dwell == {"mean": 20.0, "std": 1.5, "clip_low": 4.0, "clip_high": 25.0}
    for doc in _types(raised).values():
        assert doc["sequences"]["seek_thirst"]["steps"]["use"]["duration"]["mean"] == 20.0


def test_same_graph_settings_and_seed_render_identically(hospital: Result) -> None:
    again = read(HOSPITAL, GRAPH, S.load(), seed=0)
    assert render(again, HOSPITAL, "flowedit_queue") == render(hospital, HOSPITAL, "flowedit_queue")
    other = read(HOSPITAL, GRAPH, S.load(), seed=1)
    assert render(other, HOSPITAL, "flowedit_queue")["scenario.yaml"] != render(hospital, HOSPITAL, "flowedit_queue")["scenario.yaml"]


def test_settings_set_the_crowd_not_the_site() -> None:
    settings = S.load({"count": 3, "local_planner": "orca", "synthesize_needs": True, "needs": {"thirst": {"decay_rate": {"mean": 4.0}}}, "entrances": {"rate": 0.0}})
    result = read(HOSPITAL, GRAPH, settings)
    assert len(result.doc["humans"]) == 3 and not result.doc["regions"]
    doc = next(iter(_types(result).values()))
    assert doc["local_planner"] == "orca"
    assert doc["needs"]["thirst"]["decay_rate"] == {"mean": 4.0}
    assert doc["needs"]["thirst"]["initial"] == S.DEFAULTS["needs"]["*"]["initial"]
    assert doc["needs"]["visit_desk"]["decay_rate"]["mean"] == 0.5
    assert result.needs["thirst"].target == "fountain"


def test_settings_refuse_unknown_keys() -> None:
    with pytest.raises(S.SettingsError, match="unknown crowd setting 'crowd'"):
        S.load({"crowd": 3})
    with pytest.raises(S.SettingsError, match="needs.thirst: unknown keys"):
        S.need(S.load({"needs": {"thirst": {"rate": 1.0}}}), "thirst")


def test_the_exterior_door_is_a_source_and_a_sink_inside_it(hospital: Result) -> None:
    regions = hospital.doc["regions"]
    assert sorted(regions) == ["main_hallway_entrance_sink", "main_hallway_entrance_source"]
    source, sink = regions["main_hallway_entrance_source"], regions["main_hallway_entrance_sink"]
    assert all(9.25 <= p["x"] <= 10.75 and 0.0 < p["y"] <= 1.8 for p in source["polygon"] + sink["polygon"])
    assert max(p["y"] for p in sink["polygon"]) < min(p["y"] for p in source["polygon"])
    agent = source["config"]["agent"]
    assert agent["sink_affinity"] == [{"sink": "main_hallway_entrance_sink", "weight": 1.0}]
    visitor = hospital.types[agent["agent_type"]]
    exit_pose = visitor["sequences"]["leave"]["steps"]["exit"]["target_pose"]
    assert 9.4 <= exit_pose["x"] <= 10.6 and 0.1 <= exit_pose["y"] <= 0.6
    assert visitor["sequences"]["seek_thirst"]["then"] == "visit"


def test_a_world_object_moved_in_the_graph_is_refused(tmp_path: Path) -> None:
    scene = load(GRAPH)
    name = next(f.subject for f in scene.facts if f.verb == "at" and f.subject.startswith("obj:waiting_chair"))
    with pytest.raises(ValueError, match="is an object of world.yaml"):
        read(HOSPITAL, _edited(tmp_path, name, "at", {"x": 1.0, "y": 1.0}))


def test_builder_reads_how_people_use_objects_from_humansim() -> None:
    scene = build(HOSPITAL, SHOWCASE)

    def affords(obj: str) -> list[tuple[str, dict[str, Any]]]:
        return [(str(f.object), dict(f.params)) for f in scene.facts if f.subject == obj and f.verb == "affords"]

    assert affords("obj:reception_desk") == [("interaction:QUEUE_USE", {"from": "patient.yaml#default.check_in"})]
    assert ("interaction:TALK_TO", {"from": "doctor.yaml#default.consult", "meet": True}) in affords("obj:exam_chair_1")
    assert ("interaction:SERVICE", {"from": "nurse.yaml#leg_exam.offer", "meet": True, "tag": "nurse_escort", "offer": True}) in affords("obj:exam_chair_1")
    ((kind, params),) = affords("obj:patient_bed_1")
    assert kind == "interaction:SERVICE" and {k: v for k, v in params.items() if k != "from"} == {"meet": True, "tag": "nurse_escort", "seek": True, "offer": True}
    satisfies = {(str(f.object), f.params.get("from")) for f in scene.facts if f.subject == "obj:reception_desk" and f.verb == "satisfies"}
    assert satisfies == {("need:checkin", "patient.yaml#default.check_in")}


def test_service_objects_are_not_sought_and_step_needs_are(tmp_path: Path) -> None:
    scene = build(HOSPITAL, SHOWCASE)
    path = tmp_path / "flowgraph.yaml"
    write_kg(scene, path)
    notes: list[str] = []
    result = read(HOSPITAL, path, S.load({"count": 2, "synthesize_needs": False}), notes=notes)
    assert result.needs["checkin"].target == "reception_desk" and result.needs["checkin"].kind == "QUEUE_USE"
    assert (result.needs["treatment"].target, result.needs["treatment"].kind, result.needs["treatment"].meet) == ("exam_chair_1", "TALK_TO", True)
    assert all("patient_bed_1" not in p.objects for p in result.needs.values())


@pytest.mark.parametrize(
    ("block", "model", "expected"),
    [
        ({"type": "patient_bed", "capacity": 1}, "Hospital/Isolation_Bed", ("LIE_ON", False, "rule:prone")),
        ({"type": "chair", "capacity": 1}, "", ("SIT_ON", False, "rule:seated")),
        ({"type": "spot", "seats": [{"x": 0.0, "y": 0.0}]}, "", ("SIT_ON", False, "rule:seated")),
        ({"type": "desk", "capacity": 3, "formation": {"type": "line"}}, "Hospital/Reception_Counter", ("QUEUE_USE", False, "rule:formation line")),
        ({"type": "gathering_area", "capacity": 0, "formation": {"type": "f_formation"}}, "Hospital/Waiting_Room_Bench", ("GROUP_CONVERSATION", True, "rule:formation f_formation")),
        ({"type": "table", "capacity": 1}, "Hospital/Bedside_Table", ("USE", False, "rule:capacity")),
        ({"type": "exit"}, "Common/Object/painting_1", None),
    ],
)
def test_rules_follow_humansims_registry(block: dict[str, Any], model: str, expected: tuple[str, bool, str] | None) -> None:
    assert by_rule(block, model) == expected


def _world_copy(tmp_path: Path) -> Path:
    root = tmp_path / "hospital_1"
    shutil.copytree(HOSPITAL, root)
    return root


def test_write_replaces_only_generated_scenarios(tmp_path: Path) -> None:
    root = _world_copy(tmp_path)
    result = read(root, root / "scenarios" / "queue" / "flowgraph.yaml", S.load({"count": 2}))
    out = write(result, root, "flowedit_queue")
    assert sorted(p.name for p in out.iterdir()) == ["provenance.yaml", "scenario.yaml", "types"]
    assert len(list((out / "types").glob("*.yaml"))) == 3
    smaller = read(root, root / "scenarios" / "queue" / "flowgraph.yaml", S.load({"count": 1}))
    write(smaller, root, "flowedit_queue")
    assert len(list((out / "types").glob("*.yaml"))) == 2
    with pytest.raises(WouldOverwrite):
        write(result, root, "queue")
    provenance = yaml.safe_load((out / "provenance.yaml").read_text())
    assert provenance["graph"] == "scenarios/queue/flowgraph.yaml" and provenance["needs"]["thirst"]["objects"] == ["fountain"]


def test_cli_reads_a_graph_into_a_scenario(tmp_path: Path) -> None:
    root = _world_copy(tmp_path)
    assert main(["read", "hospital_1", "queue", "--worlds", str(tmp_path), "--count", "2", "--write"]) == 0
    text = (root / "scenarios" / "flowedit_queue" / "scenario.yaml").read_text()
    assert "name: fountain" in text and "flow_2" in text


def test_cli_names_a_missing_graph(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["read", "hospital_1", "showcase", "--worlds", str(WORLDS)]) == 1
    assert "no FLOWEDIT graph" in capsys.readouterr().err


def test_a_graph_scene_without_affords_reads_by_rule(tmp_path: Path) -> None:
    scene: Scene = load(GRAPH)
    for f in [f for f in scene.facts if f.verb == "affords"]:
        scene.remove_fact(f.subject, f.verb, f.object, f.params)
    path = tmp_path / "flowgraph.yaml"
    write_kg(scene, path)
    result = read(HOSPITAL, path, S.load({"count": 1, "synthesize_needs": True}))
    assert result.needs["thirst"].kind == "QUEUE_USE"
    assert result.needs["visit_gathering_area"].meet
