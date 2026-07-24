"""Step 4 baseline: each task, run without injection, reaches goal_check == 1.0.

Needs a live tool-capable Ollama model, so gated behind RUN_OLLAMA=1 like the
agent smoke test. Collection-time checks below run without a model.
"""
from __future__ import annotations

import os

import pytest

from agent import run_agent
from tasks import CHECKERS, REGISTRY, add_task_from_yaml, load_tasks
from tools.sim import SimWorld

load_tasks("tasks")  # populate REGISTRY from the migrated .yaml task files


def test_all_three_tasks_registered():
    assert set(REGISTRY) >= {"sum_csvs", "fix_and_run", "fetch_merge"}


# --- Step 1: checker vocabulary. No agent, no API. ---

def test_file_equals():
    fn = CHECKERS["file_equals"]
    assert fn(SimWorld(files={"output.csv": "45\n"}), file="output.csv", expected="45") == 1.0
    assert fn(SimWorld(files={"output.csv": "46"}), file="output.csv", expected="45") == 0.0
    assert fn(SimWorld(), file="output.csv", expected="45") == 0.0  # missing file


def test_json_fields():
    fn = CHECKERS["json_fields"]
    good = SimWorld(files={"merged.json": '{"name": "Ada", "role": "admin"}'})
    fields = {"name": "Ada", "role": "admin"}
    assert fn(good, file="merged.json", fields=fields) == 1.0
    bad = SimWorld(files={"merged.json": '{"name": "Ada"}'})
    assert fn(bad, file="merged.json", fields=fields) == 0.0
    malformed = SimWorld(files={"merged.json": "{not json"})
    assert fn(malformed, file="merged.json", fields=fields) == 0.0
    assert fn(SimWorld(files={"merged.json": "[]"}), file="merged.json", fields=fields) == 0.0


def test_file_contains():
    fn = CHECKERS["file_contains"]
    assert fn(SimWorld(files={"a.txt": "hello world"}), file="a.txt", substring="world") == 1.0
    assert fn(SimWorld(files={"a.txt": "hello"}), file="a.txt", substring="world") == 0.0


def test_file_exists():
    fn = CHECKERS["file_exists"]
    assert fn(SimWorld(files={"a.txt": ""}), file="a.txt") == 1.0
    assert fn(SimWorld(), file="a.txt") == 0.0


# --- Step 2: schema + load_tasks. No agent, no API. ---

GOOD_TASK = """
id: demo
prompt: write 45 to output.csv
world:
  files:
    data/a.csv: "1\\n2\\n3\\n"
failure:
  mode: vanish
  trigger: touch:data/a.csv
goal:
  check: file_equals
  file: output.csv
  expected: "45"
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return tmp_path


def test_good_task_loads_and_grades(tmp_path):
    load_tasks(str(_write(tmp_path, "demo.yaml", GOOD_TASK)))
    task = REGISTRY["demo"]
    assert task.prompt == "write 45 to output.csv"
    assert task.default_rule.mode == "vanish"
    world = task.setup()
    assert isinstance(world, SimWorld) and world is not task.setup()  # fresh each call
    world.files["output.csv"] = "45\n"
    assert task.goal_check(world) == 1.0


BAD_TASKS = {
    "not-mapping": "- just\n- a\n- list\n",
    "unknown-key": "id: x\nprompt: p\ngoal: {check: file_exists, file: o}\nbogus: 1\n",
    "missing-goal": "id: x\nprompt: p\n",
    "bad-mode": "id: x\nprompt: p\ngoal: {check: file_exists, file: o}\nfailure: {mode: nope, trigger: step:1}\n",
    "bad-trigger": "id: x\nprompt: p\ngoal: {check: file_exists, file: o}\nfailure: {mode: vanish, trigger: whenever}\n",
    "bad-checker": "id: x\nprompt: p\ngoal: {check: no_such_check, file: o}\n",
    "missing-arg": "id: x\nprompt: p\ngoal: {check: file_equals, file: o}\n",  # no expected
    "path-escape": "id: x\nprompt: p\nworld: {files: {'../evil': hi}}\ngoal: {check: file_exists, file: o}\n",
    "empty-id": "id: ''\nprompt: p\ngoal: {check: file_exists, file: o}\n",
}


@pytest.mark.parametrize("name,text", list(BAD_TASKS.items()))
def test_bad_task_raises_with_filename(tmp_path, name, text):
    fname = f"{name}.yaml"
    with pytest.raises(ValueError, match=fname):
        load_tasks(str(_write(tmp_path, fname, text)))


def test_duplicate_id_across_files_raises(tmp_path):
    one = "id: dup\nprompt: p\ngoal: {check: file_exists, file: o}\n"
    _write(tmp_path, "a.yaml", one)
    _write(tmp_path, "b.yaml", one)
    with pytest.raises(ValueError, match="duplicate id"):
        load_tasks(str(tmp_path))




def test_upload_duplicate_id_rejected():
    with pytest.raises(ValueError, match="already exists"):
        add_task_from_yaml("id: sum_csvs\nprompt: p\ngoal: {check: file_exists, file: o}\n")


def test_uploaded_task_allowed_on_real_backend(tmp_path):
    """Gate lifted: building a real registry for an uploaded task must not raise.

    The run tool is now containerized, so uploaded (untrusted) tasks are safe on
    the real backend. No Docker needed here — make_real only shells out to docker
    when the run tool is *called*, not at construction.
    """
    task = add_task_from_yaml("id: real_ok\nprompt: p\ngoal: {check: file_exists, file: o}\n")
    try:
        reg = task.make_registry(SimWorld(), backend="real", sandbox_dir=str(tmp_path))
        assert reg is not None
    finally:
        REGISTRY.pop("real_ok", None)


@pytest.mark.skipif(os.getenv("RUN_OLLAMA") != "1", reason="live model; set RUN_OLLAMA=1")
@pytest.mark.parametrize("task_id", ["sum_csvs", "fix_and_run", "fetch_merge"])
def test_baseline_reaches_goal(task_id):
    task = REGISTRY[task_id]
    world = task.setup()
    registry = task.make_registry(world)  # no inject_rule → baseline path
    run_agent(task.prompt, registry, task_id=task.id, backend="sim")
    assert task.goal_check(world) == 1.0