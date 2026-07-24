"""Task contract + registry.

A Task bundles a prompt with the world it runs in and the grader that scores
the result. Tasks share the sim tools, so make_registry needs no per-task
override — it just wraps make_sim, optionally with an injection rule.
"""
from __future__ import annotations

import glob
import inspect
import json
import os
from dataclasses import dataclass
from typing import Callable

import yaml

from injector import MODES, FailureRule, InjectingRegistry, valid_trigger
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


# --- YAML task loader -------------------------------------------------------
# A .yaml task is data conforming to one schema. Validation is the "conform to
# template" enforcement: fail loud at load, naming the offending file, so a bad
# task never reaches run_agent. See TASK_TEMPLATES.md Step 2 for the schema.

_TOP_KEYS = {"id", "prompt", "world", "failure", "goal"}
_WORLD_KEYS = {"files", "api", "commands"}


def _bad(path: str, reason: str) -> None:
    raise ValueError(f"{path}: {reason}")


def _safe_rel(p: str) -> bool:
    """Reject sandbox escapes: absolute paths, `..` segments, `~`."""
    if os.path.isabs(p) or p.startswith("~"):
        return False
    return ".." not in p.replace("\\", "/").split("/")


def _build_task(path: str, doc: object) -> Task:
    if not isinstance(doc, dict):
        _bad(path, "top-level must be a mapping")
    unknown = set(doc) - _TOP_KEYS
    if unknown:
        _bad(path, f"unknown top-level key(s): {sorted(unknown)}")
    for key in ("id", "prompt", "goal"):
        if key not in doc:
            _bad(path, f"missing required key: {key}")
    for key in ("id", "prompt"):
        if not isinstance(doc[key], str) or not doc[key].strip():
            _bad(path, f"{key} must be a non-empty string")

    world = doc.get("world") or {}
    if not isinstance(world, dict) or set(world) - _WORLD_KEYS:
        _bad(path, f"world keys must be a subset of {sorted(_WORLD_KEYS)}")
    for sub in _WORLD_KEYS:
        section = world.get(sub) or {}
        if not isinstance(section, dict):
            _bad(path, f"world.{sub} must be a mapping")
        if sub == "files":
            for fp in section:
                if not _safe_rel(fp):
                    _bad(path, f"world.files path escapes sandbox: {fp!r}")

    failure = doc.get("failure")
    rule = None
    if failure is not None:
        if not isinstance(failure, dict):
            _bad(path, "failure must be a mapping or null")
        if failure.get("mode") not in MODES:
            _bad(path, f"failure.mode {failure.get('mode')!r} not in {sorted(MODES)}")
        if not valid_trigger(failure.get("trigger", "")):
            _bad(path, f"failure.trigger not parseable: {failure.get('trigger')!r}")
        times = failure.get("times", 1)
        if not isinstance(times, int) or isinstance(times, bool):
            _bad(path, f"failure.times must be an int: {times!r}")
        rule = FailureRule(failure["mode"], failure["trigger"], times)

    goal = doc["goal"]
    if not isinstance(goal, dict) or "check" not in goal:
        _bad(path, "goal must be a mapping with a 'check' key")
    check = goal["check"]
    if check not in CHECKERS:
        _bad(path, f"goal.check {check!r} not in {sorted(CHECKERS)}")
    goal_args = {k: v for k, v in goal.items() if k != "check"}
    try:  # missing/extra checker args surface here, not at grade time
        inspect.signature(CHECKERS[check]).bind(SimWorld(), **goal_args)
    except TypeError as e:
        _bad(path, f"goal args invalid for {check!r}: {e}")

    return Task(
        id=doc["id"],
        prompt=doc["prompt"],
        setup=lambda w=world: SimWorld(**w),  # fresh world per run
        goal_check=lambda world, c=check, a=goal_args: CHECKERS[c](world, **a),
        default_rule=rule,
    )


def load_tasks(dir: str = "tasks") -> dict[str, Task]:
    """Glob *.yaml under `dir`, validate + build each, populate REGISTRY."""
    loaded: dict[str, Task] = {}
    for path in sorted(glob.glob(os.path.join(dir, "*.yaml"))):
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        task = _build_task(path, doc)
        if task.id in loaded:
            _bad(path, f"duplicate id: {task.id!r}")
        loaded[task.id] = task
    REGISTRY.update(loaded)
    return loaded
