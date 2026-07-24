# Plan-Repair Harness

Does an LLM agent **replan** when the world breaks mid-task, or does it **flail**?
This harness runs a tool-use agent on a multi-step task, injects a **deterministic
failure** partway through, and scores the resulting trajectory: did it recover, how
expensively, and did its stated plan actually change.

Agents look fine on happy-path benchmarks and fall apart under mid-execution
failure — this measures that gap.

## How it works

```
task + tools ──▶ agent loop ──▶ trajectory ──▶ analyzer ──▶ verdict: replan | partial | flail
                     ▲
              failure injector (wraps the tool registry)
```

Per task: one **baseline** run (no failure) sets the step denominator, then N
injected runs fire the task's failure rule. The analyzer diffs each injected run
against the baseline.

## Setup

Needs [Ollama](https://ollama.com) running locally with a tool-capable model:

```bash
ollama pull qwen2.5-coder
pip install -r requirements.txt
```

The agent talks to Ollama's OpenAI-compatible endpoint (`localhost:11434/v1`) — no
API key, no cost, runs offline.

## Run it

CLI:

```bash
python run.py --tasks sum_csvs fix_and_run --backend sim --seeds 3
```

Flags: `--tasks` (ids, required), `--backend sim|real` (default `sim`),
`--seeds` (injected trials per task, default 3), `--model` (default `qwen2.5-coder`).
Writes `results.json` and prints verdict tables per task and per failure mode.

Web dashboard:

```bash
python server.py          # http://localhost:8000
```

Pick tasks, set model/seeds/backend, run, and drill into any injected trajectory
turn by turn. You can also **upload a task** as YAML from the UI (see safety below).

## Tasks are YAML

A task is data conforming to one schema — no Python, validated at load. Files live
in [tasks/](tasks/) and are loaded by `load_tasks()`.

```yaml
id: sum_csvs                    # unique, non-empty
prompt: >                       # what the agent is told to do
  Sum every number across all CSVs under data/ and write the total to output.csv.
world:                          # the starting world (all sections optional)
  files: {data/a.csv: "1\n2\n3\n"}   # in-memory fake filesystem
  api:   {}                          # fake http_get endpoints
  commands: {}                       # canned `run` outputs
failure:                        # the failure to inject (omit / null for none)
  mode: vanish
  trigger: touch:data/b.csv
  times: 1                      # optional, default 1
goal:                           # how the run is graded
  check: file_equals
  file: output.csv
  expected: "45"
```

**Checkers** (`goal.check`, each returns 0.0–1.0):

| check | args | grade 1.0 when |
|-------|------|----------------|
| `file_equals`   | `file`, `expected`   | file (stripped) == expected |
| `json_fields`   | `file`, `fields`     | file parses as JSON and every key/value in `fields` matches |
| `file_contains` | `file`, `substring`  | substring in file |
| `file_exists`   | `file`               | file present |

**Failure modes** (`failure.mode`): `hard_error`, `vanish`, `corrupt`,
`permission`, `timeout`.

**Triggers** (`failure.trigger`): `step:N` (Nth dispatch), `first_call:TOOL`
(first use of a tool), `touch:RESOURCE` (any call whose args mention the resource).

Validation fails loud at load, naming the offending file: unknown/missing keys,
bad checker or mode, unparseable trigger, duplicate id, or a `world.files` path
that escapes the sandbox (`..`, absolute).

## Verdicts

- **replan** — recovered (goal met) and the plan wasn't proven unchanged.
- **partial** — recovered, but with a flail signal or an unchanged plan.
- **flail** — didn't recover.

Signals are cheap heuristics over the trajectory (blind-retry of a failed call,
non-`done` stop, pre/post-injection `plan:` diff) — no second LLM required.

## Upload safety (sim-only)

Loading a task **executes no task code** (`yaml.safe_load`, validation only), so
uploading a task to the **sim** backend is safe.

The **real** backend runs `world.commands` as an actual shell. An uploaded task +
`--backend real` would be arbitrary command execution. Uploaded tasks are therefore
flagged sim-only and refused on the real backend at `Task.make_registry` — the one
chokepoint every real run routes through. Uploads are also kept in memory only, never
written to `tasks/`, so a restart can't silently promote one to a trusted builtin.

## Layout

```
run.py            CLI: matrix of baseline + injected trials, results.json + tables
server.py         stdlib dashboard server: /, /results.json, /tasks, /run, /upload
agent.py          instrumented tool-use loop (openai SDK -> Ollama) -> Trajectory
injector.py       FailureRule, the 5 failure modes, registry middleware
analyze.py        recovery metrics, flail heuristics, verdict
trajectory.py     Trajectory / Turn / ToolResult dataclasses
tasks/            *.yaml tasks + loader/validator/checkers (__init__.py)
tools/base.py     Tool, ToolRegistry, ToolResult
tools/sim.py      in-memory fake fs + fake api (deterministic, safe to break)
tools/real.py     sandboxed real shell + fs (temp dir)
web/index.html    single-page dashboard
tests/            pytest suite
```

## Tests

```bash
pytest                    # unit + collection tests
RUN_OLLAMA=1 pytest       # also run the live-model baseline/E2E tests
```
