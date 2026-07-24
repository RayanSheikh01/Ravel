"""Task contract + registry.

A Task bundles a prompt with the world it runs in and the grader that scores
the result. Tasks share the sim tools, so make_registry needs no per-task
override — it just wraps make_sim, optionally with an injection rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from injector import FailureRule, InjectingRegistry
from tools.sim import SimWorld, make_sim


# Checker vocabulary: name -> fn(world, **args) -> 0.0..1.0. A checker sees only
# the final SimWorld; keep them pure, no trajectory access.
def _file_equals(world: SimWorld, file: str, expected: str) -> float:
    return 1.0 if world.files.get(file, "").strip() == str(expected).strip() else 0.0


def _json_fields(world: SimWorld, file: str, fields: dict) -> float:
    try:
        data = json.loads(world.files.get(file, ""))
    except (json.JSONDecodeError, TypeError):
        return 0.0
    if not isinstance(data, dict):
        return 0.0
    return 1.0 if all(data.get(k) == v for k, v in fields.items()) else 0.0


def _file_contains(world: SimWorld, file: str, substring: str) -> float:
    return 1.0 if substring in world.files.get(file, "") else 0.0


def _file_exists(world: SimWorld, file: str) -> float:
    return 1.0 if file in world.files else 0.0


CHECKERS: dict[str, Callable[..., float]] = {
    "file_equals": _file_equals,
    "json_fields": _json_fields,
    "file_contains": _file_contains,
    "file_exists": _file_exists,
}


@dataclass
class Task:
    id: str
    prompt: str
    setup: Callable[[], SimWorld]          # fresh world for one run
    goal_check: Callable[[SimWorld], float]  # 0.0..1.0 grade after the run
    default_rule: FailureRule | None = None  # the failure that makes sense here

    def make_registry(
        self,
        world: SimWorld,
        *,
        inject_rule: FailureRule | None = None,
        backend: str = "sim",
        sandbox_dir: str | None = None,
    ) -> InjectingRegistry:
        if backend == "real":
            from tools.real import make_real  # local import: real backend is optional
            if sandbox_dir is None:
                raise ValueError("backend='real' needs sandbox_dir")
            inner = make_real(world, sandbox_dir)
        else:
            inner = make_sim(world)
        return InjectingRegistry(inner, inject_rule)


REGISTRY: dict[str, Task] = {}


def register(task: Task) -> Task:
    REGISTRY[task.id] = task
    return task
