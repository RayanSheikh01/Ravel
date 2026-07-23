import json


def plan_changed(before, after):
    """True if the plan changed between two states."""
    return before != after


class Metrics:
    """Collects metrics from a run, and computes a verdict."""

    def __init__(self, success, recovery_steps, recovery_ratio, flail, plan_changed, verdict=None):
        self.success = success
        self.recovery_steps = recovery_steps
        self.recovery_ratio = recovery_ratio
        self.flail = flail
        self.plan_changed = plan_changed
        self.verdict = verdict if verdict is not None else self._compute_verdict()

    def _compute_verdict(self):
        if self.flail:
            return "flail"
        if self.plan_changed:
            return "replan"
        return "success" if self.success else "fail"


def _blind_retry(traj, k=2):
    """Same (tool, args) called >=k times after it already returned ok=False."""
    failed, counts = set(), {}
    for c in traj.all_calls():
        key = (c.tool, json.dumps(c.args, sort_keys=True))
        if key in failed:
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= k:
                return True
        if c.result and not c.result.ok:
            failed.add(key)
    return False


def _plan_lines(traj, lo, hi):
    return [
        ln.strip()
        for t in traj.turns[lo:hi]
        for ln in t.assistant_text.splitlines()
        if ln.strip().lower().startswith("plan:")
    ]


def plan_changed_heuristic(traj):
    """Diff the last plan: line before vs after injection. None if not judgeable."""
    if traj.injection_step is None:
        return None
    before = _plan_lines(traj, 0, traj.injection_step)
    after = _plan_lines(traj, traj.injection_step, len(traj.turns))
    if not before or not after:
        return None
    return after[-1] != before[-1]


def analyze(traj, baseline_steps, success):
    """Trajectory + baseline denominator + goal grade -> Metrics with a verdict.

    Only meaningful for injected runs. Verdict table lives here (see IMPLEMENTATION
    step 5): flail = failed or a flag fired without recovering; replan = recovered
    cleanly with the plan not proven unchanged; partial = everything between.
    """
    injected = traj.injection_step is not None
    recovery_steps = (len(traj.turns) - traj.injection_step) if injected else 0
    recovery_ratio = recovery_steps / baseline_steps if baseline_steps else 0.0
    recovered = success >= 1.0
    flail_flag = _blind_retry(traj) or (injected and traj.stop_reason != "done")
    pc = plan_changed_heuristic(traj)

    if not recovered:
        verdict = "flail"
    elif flail_flag or pc is False:
        verdict = "partial"
    else:
        verdict = "replan"

    return Metrics(
        success=recovered,
        recovery_steps=recovery_steps,
        recovery_ratio=recovery_ratio,
        flail=flail_flag,
        plan_changed=pc,
        verdict=verdict,
    )
