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
