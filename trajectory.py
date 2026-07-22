from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    content: str                      # what the model sees
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]
    result: ToolResult | None = None  # filled after dispatch
    injected: bool = False            # True if the injector altered this result


@dataclass
class Turn:
    step: int
    assistant_text: str               # model prose this turn (incl. "plan:" line)
    calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Trajectory:
    task_id: str
    backend: str
    turns: list[Turn] = field(default_factory=list)
    injection_step: int | None = None  # step a failure fired; None = baseline
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""              # "done" | "max_steps" | "giveup"

    def all_calls(self) -> list[ToolCall]:
        return [c for t in self.turns for c in t.calls]
