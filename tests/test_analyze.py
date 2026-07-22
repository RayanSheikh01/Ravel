import pytest


def test_flail_and_replan(monkeypatch):
    from analyze import Metrics
    monkeypatch.setattr("analyze.plan_changed", lambda *a, **k: True)
    flail = Metrics(success=False, recovery_steps=3, recovery_ratio=0.0, flail=True, plan_changed=True, verdict=None)
    replan = Metrics(success=True, recovery_steps=1, recovery_ratio=1.0, flail=False, plan_changed=True, verdict=None)
    assert flail.verdict == "flail"
    assert replan.verdict == "replan"