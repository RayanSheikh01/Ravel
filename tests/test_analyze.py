import pytest


def test_flail_and_replan(monkeypatch):
    from analyze import Metrics
    monkeypatch.setattr("analyze.plan_changed", lambda *a, **k: True)
    flail = Metrics(success=False, recovery_steps=3, recovery_ratio=0.0, flail=True, plan_changed=True, verdict=None)
    replan = Metrics(success=True, recovery_steps=1, recovery_ratio=1.0, flail=False, plan_changed=True, verdict=None)
    assert flail.verdict == "flail"
    assert replan.verdict == "replan"


def _traj(turns, injection_step, stop_reason="done"):
    from trajectory import Trajectory
    t = Trajectory(task_id="t", backend="sim")
    t.turns = turns
    t.injection_step = injection_step
    t.stop_reason = stop_reason
    return t


def _turn(step, text, calls=()):
    from trajectory import Turn
    return Turn(step=step, assistant_text=text, calls=list(calls))


def _call(tool, args, ok):
    from trajectory import ToolCall, ToolResult
    return ToolCall(tool=tool, args=args, result=ToolResult(ok=ok, content=""))


def test_analyze_flail_vs_replan():
    from analyze import analyze

    # blind-retries a failed call 3x, never recovers -> flail
    bad = _call("read_file", {"path": "b.csv"}, ok=False)
    good = _call("read_file", {"path": "b.csv"}, ok=True)  # same sig repeated after fail
    flail_traj = _traj(
        [_turn(0, "plan: read", [bad]),
         _turn(1, "plan: read", [_call("read_file", {"path": "b.csv"}, ok=False)]),
         _turn(2, "plan: read", [_call("read_file", {"path": "b.csv"}, ok=False)])],
        injection_step=0, stop_reason="max_steps",
    )
    assert analyze(flail_traj, baseline_steps=3, success=0.0).verdict == "flail"

    # hits failure then a different successful call, revised plan -> replan
    replan_traj = _traj(
        [_turn(0, "plan: read b.csv", [bad]),
         _turn(1, "plan: skip b.csv, read a.csv instead",
                [_call("read_file", {"path": "a.csv"}, ok=True)])],
        injection_step=0, stop_reason="done",
    )
    assert analyze(replan_traj, baseline_steps=2, success=1.0).verdict == "replan"