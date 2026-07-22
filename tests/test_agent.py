"""Step 2 test: a trivial task on the sim, no injector, finishes `done`
and states a plan. Requires Ollama running with qwen2.5-coder pulled."""
from __future__ import annotations

from agent import run_agent
from tools.sim import SimWorld, make_sim


def test_trivial_task_finishes_done_with_plan():
    world = SimWorld(files={"note.txt": "hello"})
    registry = make_sim(world)
    traj = run_agent(
        "Read the file 'note.txt' and tell me its contents.",
        registry,
        task_id="trivial",
        backend="sim",
    )
    assert traj.stop_reason == "done"
    assert any("plan:" in t.assistant_text.lower() for t in traj.turns)
