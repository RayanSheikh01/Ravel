from __future__ import annotations

from dataclasses import dataclass, field

from tools.base import Tool, ToolRegistry
from trajectory import ToolResult


@dataclass
class SimWorld:
    """In-memory world the sim tools share. Deterministic, safe to break."""
    files: dict[str, str] = field(default_factory=dict)
    api: dict[str, str] = field(default_factory=dict)       # fake http_get endpoints
    commands: dict[str, str] = field(default_factory=dict)  # canned `run` outputs


class ReadFile(Tool):
    def __init__(self, world: SimWorld):
        super().__init__("read_file", "Read a file's contents.",
                         {"type": "object",
                          "properties": {"path": {"type": "string"}},
                          "required": ["path"]})
        self.world = world

    def call(self, args) -> ToolResult:
        path = args["path"]
        if path not in self.world.files:
            return ToolResult(False, f"no such file: {path}")
        return ToolResult(True, self.world.files[path])


class WriteFile(Tool):
    def __init__(self, world: SimWorld):
        super().__init__("write_file", "Write (or overwrite) a file.",
                         {"type": "object",
                          "properties": {"path": {"type": "string"},
                                         "content": {"type": "string"}},
                          "required": ["path", "content"]})
        self.world = world

    def call(self, args) -> ToolResult:
        self.world.files[args["path"]] = args["content"]
        return ToolResult(True, f"wrote {args['path']}")


class ListDir(Tool):
    def __init__(self, world: SimWorld):
        super().__init__("list_dir", "List files under a path prefix.",
                         {"type": "object",
                          "properties": {"prefix": {"type": "string"}},
                          "required": ["prefix"]})
        self.world = world

    def call(self, args) -> ToolResult:
        prefix = args["prefix"]
        hits = sorted(p for p in self.world.files if p.startswith(prefix))
        return ToolResult(True, "\n".join(hits))


class Run(Tool):
    # ponytail: dumb command lookup, no real shell. Tasks register the exact
    # commands they need in world.commands. Swap for a real interpreter only if
    # a task genuinely needs one.
    def __init__(self, world: SimWorld):
        super().__init__("run", "Run a shell command.",
                         {"type": "object",
                          "properties": {"cmd": {"type": "string"}},
                          "required": ["cmd"]})
        self.world = world

    def call(self, args) -> ToolResult:
        cmd = args["cmd"]
        if cmd not in self.world.commands:
            return ToolResult(False, f"command not found: {cmd}")
        return ToolResult(True, self.world.commands[cmd])


class HttpGet(Tool):
    def __init__(self, world: SimWorld):
        super().__init__("http_get", "GET a fake API endpoint.",
                         {"type": "object",
                          "properties": {"url": {"type": "string"}},
                          "required": ["url"]})
        self.world = world

    def call(self, args) -> ToolResult:
        url = args["url"]
        if url not in self.world.api:
            return ToolResult(False, f"404 not found: {url}")
        return ToolResult(True, self.world.api[url])


def make_sim(world: SimWorld) -> ToolRegistry:
    return ToolRegistry([
        ReadFile(world), WriteFile(world), ListDir(world),
        Run(world), HttpGet(world),
    ])
