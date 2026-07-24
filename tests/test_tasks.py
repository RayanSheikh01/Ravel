"""Step 4 baseline: each task, run without injection, reaches goal_check == 1.0.

Needs a live tool-capable Ollama model, so gated behind RUN_OLLAMA=1 like the
agent smoke test. Collection-time checks below run without a model.
"""
from __future__ import annotations

import os

import pytest

from agent import run_agent
from tasks import CHECKERS, REGISTRY
from tools.sim import SimWorld
import tasks.build_report  # noqa: F401 — import registers the task
import tasks.fix_and_run   # noqa: F401
import tasks.fetch_merge   # noqa: F401


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


@pytest.mark.skipif(os.getenv("RUN_OLLAMA") != "1", reason="live model; set RUN_OLLAMA=1")
@pytest.mark.parametrize("task_id", ["sum_csvs", "fix_and_run", "fetch_merge"])
def test_baseline_reaches_goal(task_id):
    task = REGISTRY[task_id]
    world = task.setup()
    registry = task.make_registry(world)  # no inject_rule → baseline path
    run_agent(task.prompt, registry, task_id=task.id, backend="sim")
    assert task.goal_check(world) == 1.0
