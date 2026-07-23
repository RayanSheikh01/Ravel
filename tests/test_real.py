"""Step 7: real tools hit disk; run shells out. Direct-dispatch test needs no
model. The end-to-end via run_agent is gated behind RUN_OLLAMA=1 like the others.
"""
from __future__ import annotations

import os
import sys

import pytest

from tools.real import make_real
from tools.sim import SimWorld


def test_real_tools_hit_disk_and_run(tmp_path):
    world = SimWorld(files={"seed.txt": "hi\n"})
    reg = make_real(world, str(tmp_path))

    # make_real seeds the sandbox from world.files
    assert (tmp_path / "seed.txt").read_text() == "hi\n"

    # write hits real disk and mirrors into world.files (for goal_check)
    reg.dispatch("write_file", {"path": "out.txt", "content": "abc"})
    assert (tmp_path / "out.txt").read_text() == "abc"
    assert world.files["out.txt"] == "abc"

    # read comes off disk
    r = reg.dispatch("read_file", {"path": "out.txt"})
    assert r.ok and r.content == "abc"

    # list walks real files
    r = reg.dispatch("list_dir", {"prefix": ""})
    assert {"out.txt", "seed.txt"} <= set(r.content.split())

    # run executes a real subprocess and returns its stdout
    r = reg.dispatch("run", {"cmd": f'"{sys.executable}" -c "print(6*7)"'})
    assert r.ok and "42" in r.content

    # path escape is refused (dispatch turns the raise into ok=False)
    r = reg.dispatch("read_file", {"path": "../escape.txt"})
    assert not r.ok


@pytest.mark.skipif(os.getenv("RUN_OLLAMA") != "1", reason="live model; set RUN_OLLAMA=1")
def test_fix_and_run_on_real_backend(tmp_path):
    from agent import run_agent
    from tasks import REGISTRY
    import tasks.fix_and_run  # noqa: F401 — registers the task

    task = REGISTRY["fix_and_run"]
    world = task.setup()
    reg = task.make_registry(world, backend="real", sandbox_dir=str(tmp_path))
    run_agent(task.prompt, reg, task_id=task.id, backend="real")
    assert task.goal_check(world) == 1.0
