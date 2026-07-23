from __future__ import annotations

import os
import subprocess

from tools.base import Tool, ToolRegistry
from tools.sim import HttpGet, SimWorld
from trajectory import ToolResult

RUN_TIMEOUT = 30  # ponytail: fixed cap; make it a task/CLI knob only if a task needs longer


def _safe(sandbox_dir: str, path: str) -> str:
    """Resolve path under sandbox_dir, refusing anything that escapes it."""
    full = os.path.normpath(os.path.join(sandbox_dir, path))
    if os.path.commonpath([sandbox_dir, full]) != sandbox_dir:
        raise ValueError(f"path escapes sandbox: {path}")
    return full


class RealReadFile(Tool):
    def __init__(self, world: SimWorld, sandbox_dir: str):
        super().__init__("read_file", "Read a file's contents.",
                         {"type": "object",
                          "properties": {"path": {"type": "string"}},
                          "required": ["path"]})
        self.world, self.dir = world, sandbox_dir

    def call(self, args) -> ToolResult:
        full = _safe(self.dir, args["path"])
        if not os.path.isfile(full):
            return ToolResult(False, f"no such file: {args['path']}")
        with open(full, encoding="utf-8") as f:
            content = f.read()
        self.world.files[args["path"]] = content  # mirror for goal_check
        return ToolResult(True, content)


class RealWriteFile(Tool):
    def __init__(self, world: SimWorld, sandbox_dir: str):
        super().__init__("write_file", "Write (or overwrite) a file.",
                         {"type": "object",
                          "properties": {"path": {"type": "string"},
                                         "content": {"type": "string"}},
                          "required": ["path", "content"]})
        self.world, self.dir = world, sandbox_dir

    def call(self, args) -> ToolResult:
        full = _safe(self.dir, args["path"])
        os.makedirs(os.path.dirname(full) or self.dir, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(args["content"])
        self.world.files[args["path"]] = args["content"]  # mirror for goal_check
        return ToolResult(True, f"wrote {args['path']}")


class RealListDir(Tool):
    def __init__(self, world: SimWorld, sandbox_dir: str):
        super().__init__("list_dir", "List files under a path prefix.",
                         {"type": "object",
                          "properties": {"prefix": {"type": "string"}},
                          "required": ["prefix"]})
        self.world, self.dir = world, sandbox_dir

    def call(self, args) -> ToolResult:
        prefix = args["prefix"]
        hits = []
        for root, _, names in os.walk(self.dir):
            for n in names:
                rel = os.path.relpath(os.path.join(root, n), self.dir).replace(os.sep, "/")
                if rel.startswith(prefix):
                    hits.append(rel)
        return ToolResult(True, "\n".join(sorted(hits)))


class RealRun(Tool):
    def __init__(self, world: SimWorld, sandbox_dir: str):
        super().__init__("run", "Run a shell command in the sandbox.",
                         {"type": "object",
                          "properties": {"cmd": {"type": "string"}},
                          "required": ["cmd"]})
        self.world, self.dir = world, sandbox_dir

    def call(self, args) -> ToolResult:
        # shell=True is deliberate: this tool exists to execute the agent's own
        # commands inside its sandbox. Confined to sandbox_dir and timeout-capped.
        try:
            proc = subprocess.run(args["cmd"], shell=True, cwd=self.dir,
                                  capture_output=True, text=True, timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "error: timed out")
        out = (proc.stdout or "") + (proc.stderr or "")
        return ToolResult(proc.returncode == 0, out or f"exit {proc.returncode}")


def make_real(world: SimWorld, sandbox_dir: str) -> ToolRegistry:
    """5 real tools over sandbox_dir; seeds the dir from world.files first."""
    sandbox_dir = os.path.abspath(sandbox_dir)
    os.makedirs(sandbox_dir, exist_ok=True)
    for path, content in world.files.items():
        full = _safe(sandbox_dir, path)
        os.makedirs(os.path.dirname(full) or sandbox_dir, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    return ToolRegistry([
        RealReadFile(world, sandbox_dir),
        RealWriteFile(world, sandbox_dir),
        RealListDir(world, sandbox_dir),
        RealRun(world, sandbox_dir),
        HttpGet(world),  # fake API; real network out of scope
    ])
