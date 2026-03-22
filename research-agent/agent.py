"""Custom ReAct loop using the OpenAI SDK directly.

Replaces LangGraph's create_react_agent with a hand-rolled loop:
  Reason (LLM decides) -> Act (call tool) -> Observe (feed result back) -> repeat

Provides two variants:
  - run_agent_turn()          — returns final answer string (for CLI)
  - run_agent_turn_streaming() — yields events as they happen (for web UI SSE)
"""

import json
import logging

from openai import OpenAI

from config import Settings
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

logger = logging.getLogger("research_agent")


def create_client(settings: Settings) -> OpenAI:
    """Create an OpenAI client from settings."""
    return OpenAI(
        api_key=settings.api_key.get_secret_value(),
        base_url=settings.base_url,
    )


def _execute_tool_call(tool_call) -> tuple[str, str, dict, str]:
    """Execute a single tool call and return (name, call_id, args_dict, result).

    Never raises — tool errors are returned as strings for the LLM to handle.
    """
    name = tool_call.function.name
    call_id = tool_call.id
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return (
            name,
            call_id,
            {},
            f"Error: could not parse arguments: {tool_call.function.arguments}",
        )

    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return name, call_id, args, f"Error: unknown tool '{name}'"

    try:
        result = func(**args)
    except Exception as e:
        result = f"Error executing {name}: {e}"

    return name, call_id, args, result


def run_agent_turn(
    user_input: str,
    messages: list[dict],
    settings: Settings,
    client: OpenAI,
) -> tuple[str, dict]:
    """Run one full ReAct turn (may involve multiple tool calls).

    Args:
        user_input: The user's message.
        messages: Conversation history (mutated in place).
        settings: App settings.
        client: OpenAI client instance.

    Returns:
        (final_answer, usage_totals) where usage_totals = {input, output, total}.
    """
    messages.append({"role": "user", "content": user_input})

    usage_totals = {"input": 0, "output": 0, "total": 0}

    for _iteration in range(settings.max_iterations):
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        # Track token usage
        if response.usage:
            usage_totals["input"] += response.usage.prompt_tokens
            usage_totals["output"] += response.usage.completion_tokens
            usage_totals["total"] += response.usage.total_tokens
            logger.info(
                "Tokens — input: %d, output: %d, total: %d",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
            )

        # Serialize assistant message into the history.
        # Must include tool_calls if present so the API can match tool results.
        msg_dict = {"role": "assistant"}
        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
        messages.append(msg_dict)

        # No tool calls → final answer
        if not assistant_msg.tool_calls:
            return assistant_msg.content or "", usage_totals

        # Execute each tool call and append results
        for tc in assistant_msg.tool_calls:
            name, call_id, args, result = _execute_tool_call(tc)
            logger.info(
                "Tool [%s](%s): %s", name, json.dumps(args, ensure_ascii=False)[:200], result[:300]
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                }
            )

    # Loop exhausted without a final answer
    return (
        "I've reached the maximum number of reasoning steps. "
        "Here's what I found so far — please try a more specific query.",
        usage_totals,
    )


def run_agent_turn_streaming(
    user_input: str,
    messages: list[dict],
    settings: Settings,
    client: OpenAI,
):
    """Streaming variant of run_agent_turn — yields events for the web UI.

    Yields dicts with these types:
      {"type": "tool_call", "name": ..., "args": ..., "call_id": ...}
      {"type": "tool_result", "name": ..., "args": ..., "result": ...}
      {"type": "message", "content": ...}
      {"type": "tokens", "data": {"input": ..., "output": ..., "total": ...}}
      {"type": "done"}
    """
    messages.append({"role": "user", "content": user_input})

    usage_totals = {"input": 0, "output": 0, "total": 0}

    for _iteration in range(settings.max_iterations):
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        # Track token usage
        if response.usage:
            usage_totals["input"] += response.usage.prompt_tokens
            usage_totals["output"] += response.usage.completion_tokens
            usage_totals["total"] += response.usage.total_tokens
            yield {"type": "tokens", "data": usage_totals.copy()}
            logger.info(
                "Tokens — input: %d, output: %d, total: %d",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
            )

        # Serialize assistant message into history
        msg_dict = {"role": "assistant"}
        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
        messages.append(msg_dict)

        # No tool calls → final answer
        if not assistant_msg.tool_calls:
            if assistant_msg.content:
                yield {"type": "message", "content": assistant_msg.content}
            yield {"type": "done"}
            return

        # Execute each tool call
        for tc in assistant_msg.tool_calls:
            name, call_id, args, result = _execute_tool_call(tc)
            logger.info(
                "Tool [%s](%s): %s", name, json.dumps(args, ensure_ascii=False)[:200], result[:300]
            )

            # Yield tool result event for the UI
            yield {
                "type": "tool_result",
                "name": name,
                "args": args,
                "result": result,
            }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                }
            )

    # Loop exhausted
    exhaust_msg = (
        "I've reached the maximum number of reasoning steps. "
        "Here's what I found so far — please try a more specific query."
    )
    yield {"type": "message", "content": exhaust_msg}
    yield {"type": "done"}
