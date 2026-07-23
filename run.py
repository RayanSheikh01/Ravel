"""Step 6 runner: baseline + injected trials per task, verdict table + results.json.

Per task: one baseline run (no rule) gives the step denominator, then `seeds`
injected trials with the task's default_rule. Wire per run:
  world -> task.make_registry(world, inject_rule) -> run_agent -> goal_check -> analyze.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import tempfile
from collections import Counter

from agent import run_agent
from analyze import analyze
from tasks import REGISTRY

# Import registers the tasks into REGISTRY.
import tasks.build_report  # noqa: F401
import tasks.fix_and_run   # noqa: F401
import tasks.fetch_merge   # noqa: F401


def _run(task, rule, *, model, backend):
    """One agent run; returns (trajectory, goal_grade). Fresh sandbox per real run."""
    world = task.setup()
    sandbox = tempfile.mkdtemp(prefix="ravel_sbx_") if backend == "real" else None
    try:
        reg = task.make_registry(world, inject_rule=rule, backend=backend, sandbox_dir=sandbox)
        traj = run_agent(task.prompt, reg, task_id=task.id, backend=backend, model=model)
        traj.injection_step = reg.injection_step  # copy off the injecting registry
        return traj, task.goal_check(world)
    finally:
        if sandbox:
            shutil.rmtree(sandbox, ignore_errors=True)


def run_tasks(task_ids, seeds, model, backend):
    rows = []
    by_task = {}   # task_id -> Counter(verdict)
    by_mode = {}   # failure mode -> Counter(verdict)

    for task_id in task_ids:
        task = REGISTRY.get(task_id)
        if task is None:
            print(f"skip: task '{task_id}' not registered")
            continue

        base, base_grade = _run(task, None, model=model, backend=backend)
        baseline_steps = max(1, len(base.turns))
        rows.append({"task": task_id, "kind": "baseline", "success": base_grade,
                     "steps": len(base.turns), "verdict": "baseline", "trajectory": [dataclasses.asdict(t) for t in base.turns]})

        mode = task.default_rule.mode if task.default_rule else "none"
        by_task.setdefault(task_id, Counter())
        by_mode.setdefault(mode, Counter())

        for i in range(seeds):
            traj, grade = _run(task, task.default_rule, model=model, backend=backend)
            m = analyze(traj, baseline_steps, grade)
            by_task[task_id][m.verdict] += 1
            by_mode[mode][m.verdict] += 1
            rows.append({
                "task": task_id, "kind": "injected", "seed": i, "mode": mode,
                "success": grade, "steps": len(traj.turns),
                "injection_step": traj.injection_step,
                "recovery_steps": m.recovery_steps, "recovery_ratio": round(m.recovery_ratio, 3),
                "flail": m.flail, "plan_changed": m.plan_changed, "verdict": m.verdict,
                "trajectory": [dataclasses.asdict(t) for t in traj.turns],
            })
            
    with open("results.json", "w") as f:
            json.dump(rows, f, indent=2)

    return rows, by_task, by_mode

def main():
    ap = argparse.ArgumentParser(description="Plan-repair harness runner.")
    ap.add_argument("--tasks", nargs="+", required=True, help="Task ids to run.")
    ap.add_argument("--backend", default="sim", help="Backend label (sim only for now).")
    ap.add_argument("--seeds", type=int, default=3, help="Injected trials per task.")
    ap.add_argument("--model", default="qwen2.5-coder", help="Ollama model.")
    args = ap.parse_args()
    
    # baseline
    rows, by_task, by_mode = run_tasks(
        task_ids=args.tasks,
        seeds=0,
        model=args.model,
        backend=args.backend
    )
    
    _print_table("Per task", by_task)
    _print_table("Per failure mode", by_mode)
    print("\nwrote results.json")
        
    rows, by_task, by_mode = run_tasks(
        task_ids=args.tasks,
        seeds=args.seeds,
        model=args.model,
        backend=args.backend
    )

    with open("results.json", "w") as f:
        json.dump(rows, f, indent=2)

    _print_table("Per task", by_task)
    _print_table("Per failure mode", by_mode)
    print("\nwrote results.json")


def _print_table(title, buckets):
    cols = ["replan", "partial", "flail"]
    print(f"\n{title}")
    print(f"  {'':16} " + " ".join(f"{c:>8}" for c in cols))
    for key, counts in buckets.items():
        print(f"  {key:16} " + " ".join(f"{counts.get(c, 0):>8}" for c in cols))


if __name__ == "__main__":
    main()
