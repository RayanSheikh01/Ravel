"""Step 4 baseline: each task, run without injection, reaches goal_check == 1.0.

Needs a live tool-capable Ollama model, so gated behind RUN_OLLAMA=1 like the
agent smoke test. Collection-time checks below run without a model.
"""
from __future__ import annotations

import os

import pytest

from agent import run_agent
from tasks import REGISTRY
import tasks.build_report  # noqa: F401 — import registers the task
import tasks.fix_and_run   # noqa: F401
import tasks.fetch_merge   # noqa: F401


def test_all_three_tasks_registered():
    assert set(REGISTRY) >= {"sum_csvs", "fix_and_run", "fetch_merge"}


@pytest.mark.skipif(os.getenv("RUN_OLLAMA") != "1", reason="live model; set RUN_OLLAMA=1")
@pytest.mark.parametrize("task_id", ["sum_csvs", "fix_and_run", "fetch_merge"])
def test_baseline_reaches_goal(task_id):
    task = REGISTRY[task_id]
    world = task.setup()
    registry = task.make_registry(world)  # no inject_rule → baseline path
    run_agent(task.prompt, registry, task_id=task.id, backend="sim")
    assert task.goal_check(world) == 1.0
