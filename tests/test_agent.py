"""Step 2 tests for the agent loop.

The loop logic (dispatch, message ordering, turn recording, injected flag,
token counting, max_steps) is model-independent, so it's tested against a
scripted fake client — deterministic, no Ollama needed. One live smoke test
against Ollama is opt-in via RUN_OLLAMA=1.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import agent
from agent import run_agent
from trajectory import ToolResult
from tools.base import Tool, ToolRegistry
from tools.sim import SimWorld, make_sim


def _tool_call(cid, name, arguments):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=arguments))


def _response(content, tool_calls, *, prompt=10, completion=5):
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


class FakeClient:
    """Returns a scripted list of responses, one per create() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, tools):
        self.calls.append([dict(m) if isinstance(m, dict) else m for m in messages])
        return self._responses.pop(0)


@pytest.fixture
def patch_client(monkeypatch):
    def install(responses):
        client = FakeClient(responses)
        monkeypatch.setattr(agent, "OpenAI", lambda **kw: client)
        return client
    return install


def test_loop_dispatches_tool_and_records_turn(patch_client):
    world = SimWorld()
    registry = make_sim(world)
    client = patch_client([
        _response('plan: write the file', [_tool_call("c1", "write_file",
                  '{"path": "out.txt", "content": "hi"}')], prompt=100, completion=20),
        _response("done, wrote it", None, prompt=30, completion=8),
    ])

    traj = run_agent("write out.txt", registry, task_id="t", backend="sim")

    # tool actually ran against the shared world
    assert world.files["out.txt"] == "hi"
    assert traj.stop_reason == "done"
    # one acting turn + one final turn recorded
    assert len(traj.turns) == 2
    call = traj.turns[0].calls[0]
    assert call.tool == "write_file"
    assert call.args == {"path": "out.txt", "content": "hi"}
    assert call.result.ok and not call.injected
    # tokens summed across both create() calls
    assert traj.input_tokens == 130 and traj.output_tokens == 28

    # ordering: assistant-with-tool_calls precedes its tool result on the 2nd call
    second_msgs = client.calls[1]
    roles = [m["role"] if isinstance(m, dict) else m.role for m in second_msgs]
    assert roles[-2:] == ["assistant", "tool"]


def test_injected_flag_propagates_from_meta(patch_client):
    class Injected(Tool):
        def __init__(self):
            super().__init__("boom", "d", {"type": "object", "properties": {}})

        def call(self, args):
            return ToolResult(False, "injected failure", meta={"injected": True})

    registry = ToolRegistry([Injected()])
    patch_client([
        _response("plan: call boom", [_tool_call("c1", "boom", "{}")]),
        _response("giving up", None),
    ])

    traj = run_agent("go", registry, task_id="t", backend="sim")

    assert traj.turns[0].calls[0].injected is True


def test_max_steps_stops_the_loop(patch_client):
    world = SimWorld()
    registry = make_sim(world)
    # every response asks for another tool call → never "done"
    always = [_response("plan: loop", [_tool_call(f"c{i}", "list_dir", '{"prefix": ""}')])
              for i in range(10)]
    patch_client(always)

    traj = run_agent("loop", registry, task_id="t", backend="sim", max_steps=3)

    assert traj.stop_reason == "max_steps"
    assert len(traj.turns) == 3


@pytest.mark.skipif(os.getenv("RUN_OLLAMA") != "1", reason="live model; set RUN_OLLAMA=1")
def test_live_smoke_finishes_done_with_plan():
    world = SimWorld(files={"note.txt": "hello"})
    traj = run_agent("Read note.txt and report its contents.", make_sim(world),
                     task_id="trivial", backend="sim")
    assert traj.stop_reason == "done"
    assert any("plan:" in t.assistant_text.lower() for t in traj.turns)
    

