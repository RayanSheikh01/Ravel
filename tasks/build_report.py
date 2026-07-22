"""Sum a column across CSVs → output.csv. Failure: b.csv vanishes mid-run."""
from __future__ import annotations

from injector import FailureRule
from tasks import Task, register
from tools.sim import SimWorld


def setup() -> SimWorld:
    return SimWorld(files={
        "data/a.csv": "1\n2\n3\n",
        "data/b.csv": "4\n5\n6\n",
        "data/c.csv": "7\n8\n9\n",
    })


def goal_check(world: SimWorld) -> float:
    return 1.0 if world.files.get("output.csv", "").strip() == "45" else 0.0


task = register(Task(
    id="sum_csvs",
    prompt=(
        "The data/ directory holds CSV files, each a single column of numbers. "
        "Sum every number across all files under data/ and write just the total "
        "to output.csv."
    ),
    setup=setup,
    goal_check=goal_check,
    default_rule=FailureRule("vanish", "touch:data/b.csv"),
))
