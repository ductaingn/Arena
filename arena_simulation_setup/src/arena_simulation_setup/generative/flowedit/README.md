# FLOWEDIT graph

One graph per site with each object's demand on its node: what each object is for. It is a [KG](../../../../AUTHORING.md#kg) scene (`arena-kg/1`) over `core@2`, `space@1` and [`flow@1`](../../kg/packages/flow@1.toml), kept in its own `flowgraph.yaml`, never in the world's `kg.yaml`. Edits change it, a forecaster reads it, simulators never read it. Persons stay in agent types.

```
world.yaml (geometry)          --+
scenario objects (humansim:)   --+-- build --> flowgraph.yaml --+
kg.yaml (room purposes, read)  --+                              +-- read --> scenarios/<name>/ --> HumanSim
                                         crowd settings + seed -+
```

`build` turns a hand-authored scenario into its graph, `read` turns a graph back into a scenario with no hand-written person: edit the graph (move an object, change a dwell) and read it again.

## CLI

`flowedit build <world> [<scenario>]`, or `python3 -m arena_simulation_setup.generative.flowedit.cli build ...`. A dry run prints the graph, `--write` writes it to `flowgraph.yaml` next to the scenario's `scenario.yaml` (or `--out`). Worlds resolve as in [generative/scenario](../scenario/README.md#cli).

`flowedit read <world> [<scenario>]` reads `scenarios/<scenario>/flowgraph.yaml` (or `--graph`) into `scenarios/flowedit_<scenario>/` (or `--name`). A dry run prints the scenario, `--write` writes it and replaces only a scenario carrying the generated marker unless `--force`. `--settings` names a crowd settings file, `--count`, `--scale` and `--local-planner` override it, `--seed` (default 0) keys every draw.

```bash
flowedit build hospital_1 queue --write
flowedit read hospital_1 queue --settings crowd.yaml --seed 3 --write
```

## Graph

| Node | Holds | From |
|---|---|---|
| `room:<zone>` | `has_purpose purpose:<role>` | the world KG's `has_purpose`, else the substring rules of [`zones.py`](../scenario/zones.py) as `params: {by: rule}` |
| `door:<name>` | `connects` between two rooms | derived from geometry |
| `obj:<name>` | prop `model`, `at floor:<level> {x, y, yaw, pose_ref}` | `world.yaml` objects and the scenario's `objects:` |
| `obj` demand | `serves service:<type> {capacity, interaction_radius}`, `satisfies need:<need> {amount}`, `has_formation formation:<type> {<params>, anchor_kind, anchor_ref}` | the object's `humansim:` block |
| `obj` dwell | `dwells <mean s> {std, clip_low, clip_high, from}` | the `duration` of the first agent type step whose `target` is the object's name, else its type, else of the interaction a `go_to` to it leads into |
| `obj` use | `affords interaction:<KIND> {from, meet, tag, offer, seek}`, `satisfies need:<need> {amount, from}` | [`interactions.py`](interactions.py): the agent type steps that use the object, else its `humansim:` block against HumanSim's interaction registry |

`contains` and `connects` are derived as when a world's KG loads. Unlike `kg.yaml`, the graph carries each object's `model` and pose, a copy of `world.yaml` that `flowedit build` refreshes; rooms and doors carry no geometry. Poses are in the level frame. A scenario object posed at a zone name gets the point the runtime would sample under `--seed`, and keeps the name as `pose_ref`. Dwell is read from the scenario-local agent type files (`./<type>.yaml`) its humans and regions name, and a note names other steps that give the same object another duration. `seats` and formations anchored at a pose are not in `flow@1`.

```yaml
- {subject: 'obj:fountain', verb: at, object: 'floor:0', params: {x: 11.0, y: 10.0, yaw: 1.57}}
- {subject: 'obj:fountain', verb: serves, object: 'service:fountain', params: {capacity: 1}}
- {subject: 'obj:fountain', verb: satisfies, object: 'need:thirst', params: {amount: 100.0}}
- {subject: 'obj:fountain', verb: has_formation, object: 'formation:line', params: {base_step: 1.0, front_offset: 0.8}}
- {subject: 'obj:fountain', verb: dwells, object: 5.0, params: {std: 1.5, clip_low: 4.0, clip_high: 6.0, from: 'colleague.yaml#drink.queue_and_drink'}}
```

Load and save it with `arena_simulation_setup.kg.load` and `write`.

## Interactions

`affords` says how people use an object, one fact per HumanSim `InteractionType`. A step with an object `target` and an object-bound kind (`USE`, `QUEUE_USE`, `SIT_ON`, `LIE_ON`) uses that object. A `go_to` to an object followed by a step whose kind takes no object (`GROUP_CONVERSATION`, `TALK_TO`, `HUG`, ..., or the tag-addressed `SERVICE`) uses it as a meeting point: `meet: true`, the `SERVICE` tag as `tag`, and `offer` / `seek` for the provider and seeker sides it was seen with. A need such a step satisfies becomes a `satisfies` fact with its `from`. An object with a `humansim:` block that no step uses is read by rule against HumanSim's registry: seats, or a posture word ending its type (`bed` prone, `chair`/`seat`/`sofa`/`bench`/`stool`/`couch` seated), pick the object-bound kind of that posture; else its formation picks the kind whose default formation it is, an object-bound one first (`line` is `QUEUE_USE`, `f_formation` the meeting kind `GROUP_CONVERSATION`); else a posture word of its model; else a capacity is `USE`. A block with none of these is a marker and affords nothing. `BLOCK` targets an agent and never an object.

```yaml
- {subject: 'obj:fountain', verb: affords, object: 'interaction:USE', params: {from: 'colleague.yaml#drink.queue_and_drink'}}
- {subject: 'obj:gathering_area', verb: affords, object: 'interaction:GROUP_CONVERSATION', params: {from: 'colleague.yaml#chat.hold_conversation', meet: true}}
- {subject: 'obj:patient_bed_1', verb: affords, object: 'interaction:LIE_ON', params: {from: 'rule:prone'}}
```

## Reading a graph

[`read.py`](read.py) needs the graph, the world it is for (zones, doors and furniture for geometry only), crowd settings and a seed, and is deterministic in all four.

| Graph | Scenario |
|---|---|
| objects the world does not hold (`at`, `serves`, `has_formation`, `satisfies` without `from`) | `objects:` with their `humansim:` block at the graph's point, in the map frame. A world object the graph moved is refused: a scenario cannot move it |
| `satisfies need:<n>` of an object people can use | every person's type declares `n` and preempts into `seek_<n>` once it falls below `below` |
| `affords` | the `seek_<n>` step: the kind at the object, or with `meet` a `go_to` to it and the kind beside it. `SERVICE` needs a provider and is not sought |
| `dwells` | that step's `duration`, its `patience` shifted by the dwell's upper bound so a use never runs out mid-use |
| `has_purpose` of rooms | where people start and the rooms they walk between, by populate's rules (homes screened against the robot's route) |
| a door only one room contains | `<door>_source` behind `<door>_sink` inside it: visitors enter, walk, use, and leave through the sink |

People are HumanSim behavior trees: each gets its own type (`types/flow__<name>.yaml`, extending its populate mix type) whose `idle` sequence walks free points drawn in its rooms, since a tree does not follow scenario waypoints. A need several objects serve is sought by their type (HumanSim takes the nearest) when they are all the objects of that type, else at one of them. With `synthesize_needs` (off by default), an object that satisfies no need gets `visit_<type>`, so every usable object is used; in HumanSim a crowd seeking a capacity-1 seat waits where it stands, so use it with few people or low rates. The robot comes from the settings, else the first hand-written scenario of the world whose route plans, else the two farthest zones. People walk the first level with zones. `provenance.yaml` holds the graph hash, the full settings, the plan of every need and the notes.

## Crowd settings

Global, not per site: the same file gives a crowd on any graph. Every key and its default is in [`settings.py`](settings.py); a file sets only what it changes, an unknown key is an error. A need's own `initial`, `decay_rate` or `below` replaces the `"*"` entry's whole.

```yaml
count: 20                       # people at the start (default: populate's count for the rooms, times scale)
needs:
  "*": {decay_rate: {mean: 0.5, std: 0.15, clip_low: 0.1, clip_high: 2.0}, below: 30.0}
  thirst: {decay_rate: {mean: 2.0, std: 0.5, clip_low: 1.0, clip_high: 4.0}}
local_planner: orca
entrances: {rate: 0.1, max_concurrent: 6, max_total: 30}
```

## Files

| File | Purpose |
|---|---|
| `build.py` | world + scenario to the graph |
| `interactions.py` | how an object is used, from agent type steps and HumanSim's interaction registry |
| `read.py` | graph + settings + seed to a scenario |
| `settings.py` | crowd settings and their defaults |
| `cli.py` | `flowedit build`, `flowedit read` |
