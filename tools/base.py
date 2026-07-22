from __future__ import annotations

from typing import Any

from trajectory import ToolResult


class Tool:
    """Base tool. Subclasses set name/description/schema and override call()."""

    def __init__(self, name: str, description: str, schema: dict[str, Any]):
        self.name = name
        self.description = description
        self.schema = schema  # JSON schema for the Anthropic `input_schema`

    def call(self, args: dict[str, Any]) -> ToolResult:
        raise NotImplementedError("Tool.call() must be implemented by subclasses.")


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def anthropic_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.schema}
            for t in self._tools.values()
        ]

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, content=f"Unknown tool: {name}")
        try:
            return tool.call(args)
        except Exception as e:  # a tool crash is a failed call, not a harness crash
            return ToolResult(ok=False, content=str(e))
