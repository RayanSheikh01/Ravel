"""Fix a broken script, run it, save output. Failure: first `run` hard-errors.

The sim `run` tool is canned — it returns world.commands[cmd] regardless of the
file, so "fixing" the script is the plan the agent must form, not real
execution. The seeded command is what a correct run produces.
"""
from __future__ import annotations

from injector import FailureRule
from tasks import Task, register
from tools.sim import SimWorld


def setup() -> SimWorld:
    return SimWorld(
        files={"script.py": "prnit(6 * 7)\n"},               # typo: prnit
        commands={"python script.py": "42\n"},
    )


def goal_check(world: SimWorld) -> float:
    return 1.0 if world.files.get("output.txt", "").strip() == "42" else 0.0


task = register(Task(
    id="fix_and_run",
    prompt=(
        "script.py has a bug. Read it, fix the bug by rewriting the file, then "
        "run it with the command `python script.py`. Write the command's output "
        "to output.txt."
    ),
    setup=setup,
    goal_check=goal_check,
    default_rule=FailureRule("hard_error", "first_call:run"),
))
