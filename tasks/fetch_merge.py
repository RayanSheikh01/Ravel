"""Fetch two records, merge, write merged.json. Failure: first GET is corrupted."""
from __future__ import annotations

import json

from injector import FailureRule
from tasks import Task, register
from tools.sim import SimWorld

USER_URL = "http://api/user"
ROLE_URL = "http://api/role"


def setup() -> SimWorld:
    return SimWorld(api={
        USER_URL: '{"name": "Ada"}',
        ROLE_URL: '{"role": "admin"}',
    })


def goal_check(world: SimWorld) -> float:
    try:
        merged = json.loads(world.files.get("merged.json", ""))
    except (json.JSONDecodeError, TypeError):
        return 0.0
    return 1.0 if merged.get("name") == "Ada" and merged.get("role") == "admin" else 0.0


task = register(Task(
    id="fetch_merge",
    prompt=(
        f"GET the JSON records at {USER_URL} and {ROLE_URL}, merge their fields "
        "into a single JSON object, and write it to merged.json."
    ),
    setup=setup,
    goal_check=goal_check,
    default_rule=FailureRule("corrupt", "first_call:http_get"),
))
