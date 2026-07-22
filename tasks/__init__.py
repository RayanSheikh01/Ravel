"""Task contract + registry.

A Task bundles a prompt with the world it runs in and the grader that scores
the result. Tasks share the sim tools, so make_registry needs no per-task
override — it just wraps make_sim, optionally with an injection rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from injector import FailureRule, InjectingRegistry
from tools.sim import SimWorld, make_sim


@dataclass
class Task:
    id: str
    prompt: str
    setup: Callable[[], SimWorld]          # fresh world for one run
    goal_check: Callable[[SimWorld], float]  # 0.0..1.0 grade after the run
    default_rule: FailureRule | None = None  # the failure that makes sense here

    def make_registry(
        self, world: SimWorld, *, inject_rule: FailureRule | None = None
    ) -> InjectingRegistry:
        return InjectingRegistry(make_sim(world), inject_rule)


REGISTRY: dict[str, Task] = {}


def register(task: Task) -> Task:
    REGISTRY[task.id] = task
    return task
