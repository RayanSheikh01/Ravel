from __future__ import annotations

import json

from openai import OpenAI

from tools.base import ToolRegistry
from trajectory import Trajectory, Turn, ToolCall

SYSTEM_PROMPT = (
    "You are a tool-using agent. Each turn, before you act, write one line "
    "starting with 'plan:' stating your current plan. When a tool call fails "
    "or returns something unexpected, explicitly revise your plan: line to "
    "explain what changed and what you will do instead. Use the tools to "
    "complete the task, then give a final answer with no tool calls."
)


def run_agent(
    task_prompt: str,
    registry: ToolRegistry,
    *,
    task_id: str,
    backend: str,
    model: str = "qwen2.5-coder",
    max_steps: int = 20,
) -> Trajectory:
    """Run the agent loop against `registry` until it stops calling tools.

    stop_reason: "done" if the model answered without tool calls, else
    "max_steps" if the loop exhausted.
    """
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key ignored by Ollama
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    traj = Trajectory(task_id=task_id, backend=backend)

    for step in range(max_steps):
        chat = client.chat.completions.create(
            model=model, messages=messages, tools=registry.openai_tools()
        )
        msg = chat.choices[0].message
        if chat.usage:  # map OpenAI usage into trajectory counters
            traj.input_tokens += chat.usage.prompt_tokens
            traj.output_tokens += chat.usage.completion_tokens

        if not msg.tool_calls:
            traj.turns.append(Turn(step=step, assistant_text=msg.content or ""))
            traj.stop_reason = "done"
            return traj

        # assistant message (with its tool_calls) must precede the tool results
        messages.append(msg)
        calls: list[ToolCall] = []
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)  # arguments is a JSON string
            res = registry.dispatch(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": res.content})
            calls.append(
                ToolCall(
                    tool=tc.function.name,
                    args=args,
                    result=res,
                    injected=bool(res.meta.get("injected")),
                )
            )
        traj.turns.append(Turn(step=step, assistant_text=msg.content or "", calls=calls))

    traj.stop_reason = "max_steps"
    return traj
