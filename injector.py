from __future__ import annotations

from dataclasses import dataclass

from tools.base import ToolRegistry
from trajectory import ToolResult


@dataclass
class FailureRule:
    mode: str          # key into MODES
    trigger: str       # "step:N" | "first_call:TOOL" | "touch:RESOURCE"
    times: int = 1     # how many dispatches to corrupt before going quiet


# Five failure modes: ToolResult -> ToolResult. Four turn a success into a
# specific error string; `corrupt` keeps ok=True but mangles the content.
def _hard_error(r: ToolResult) -> ToolResult:
    r.ok, r.content = False, "error: operation failed"
    return r


def _vanish(r: ToolResult) -> ToolResult:
    r.ok, r.content = False, "error: resource no longer exists"
    return r


def _permission(r: ToolResult) -> ToolResult:
    r.ok, r.content = False, "error: permission denied"
    return r


def _timeout(r: ToolResult) -> ToolResult:
    r.ok, r.content = False, "error: timed out"
    return r


def _corrupt(r: ToolResult) -> ToolResult:
    r.content = r.content + " [corrupted]"  # plausible-but-wrong, still ok=True
    return r


MODES = {
    "hard_error": _hard_error,
    "vanish": _vanish,
    "permission": _permission,
    "timeout": _timeout,
    "corrupt": _corrupt,
}


TRIGGER_KINDS = ("step", "first_call", "touch")  # the prefixes _fires understands


def valid_trigger(trigger: str) -> bool:
    """True iff `_fires` can parse this trigger string. Single source of truth
    for the loader's validation so the two never drift."""
    kind, sep, target = str(trigger).partition(":")
    if sep != ":" or not target:
        return False
    if kind == "step":
        return target.lstrip("-").isdigit()
    return kind in TRIGGER_KINDS


class InjectingRegistry:
    """Wraps a ToolRegistry. Same interface, so agent.py needs no changes.

    On each dispatch it gets the real result and, if the rule fires, replaces
    it via MODES[mode] and flags meta["injected"]. Fully deterministic — no RNG.
    """

    def __init__(self, inner: ToolRegistry, rule: FailureRule | None):
        self.inner = inner
        self.rule = rule
        self._step = 0                     # 0-indexed dispatch counter
        self._fired = 0                    # how many times we've injected
        self._seen_first_call = False      # for first_call:TOOL
        self.injection_step: int | None = None

    def openai_tools(self) -> list[dict]:
        return self.inner.openai_tools()

    def dispatch(self, name: str, args: dict) -> ToolResult:
        res = self.inner.dispatch(name, args)
        if self.rule and self._fired < self.rule.times and self._fires(name, args):
            MODES[self.rule.mode](res)
            res.meta["injected"] = True
            if self.injection_step is None:
                self.injection_step = self._step
            self._fired += 1
        self._step += 1
        return res

    def _fires(self, name: str, args: dict) -> bool:
        assert self.rule is not None  # only called when a rule is set
        kind, _, target = self.rule.trigger.partition(":")
        if kind == "step":
            # step:2 fires on the 3rd dispatch (0-indexed counter).
            return self._step == int(target)
        if kind == "first_call":
            if name == target and not self._seen_first_call:
                self._seen_first_call = True
                return True
            return False
        if kind == "touch":
            return any(target in str(v) for v in args.values())
        raise ValueError(f"unknown trigger: {self.rule.trigger}")
