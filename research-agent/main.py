"""Console REPL interface for the Research Agent.

Run with: python main.py
Uses a custom ReAct loop (no LangChain/LangGraph) with the OpenAI SDK.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

from agent import create_client, run_agent_turn_streaming
from config import APP_VERSION, SYSTEM_PROMPT, Settings

Path("logs").mkdir(exist_ok=True)

# Dual logging: console + rotating file (5 MB, 3 backups)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler("logs/agent.log", maxBytes=5_000_000, backupCount=3),
    ],
)
logger = logging.getLogger("research_agent")

settings = Settings()


def _get_tool_call_args(name: str, args: dict) -> str:
    """Extract the primary argument from a tool call for display."""
    if name == "web_search":
        return args.get("query", "")
    if name == "read_url":
        return args.get("url", "")
    if name == "write_report":
        return args.get("description", "")
    if name == "read_file":
        return args.get("filename", "")
    if name == "knowledge_search":
        return args.get("query", "")
    return ""


def _format_tool_status(name: str, args: dict, result: str) -> str:
    """Format a tool result into a human-friendly one-liner for the console."""
    primary_arg = _get_tool_call_args(name, args)
    args_part = f'("{primary_arg}") ' if primary_arg else " "

    if name == "knowledge_search":
        count = (
            result.count("---") + 1 if "---" in result else (0 if "No relevant" in result else 1)
        )
        return f"  [knowledge_search]{args_part}— {count} documents found"
    if name == "web_search":
        count = result.count("Title:")
        return f"  [web_search]{args_part}— {count} results found"
    if name == "read_url":
        if result.startswith("Error"):
            return f"  [read_url]{args_part}— {result[:80]}"
        return f"  [read_url]{args_part}— extracted {len(result):,} chars"
    if name == "write_report":
        return f"  [write_report]{args_part}— {result}"
    if name == "list_reports":
        count = result.count(". ")
        return f"  [list_reports] — {count} reports found"
    if name == "read_file":
        if result.startswith("Error"):
            return f"  [read_file]{args_part}— {result[:80]}"
        return f"  [read_file]{args_part}— {len(result):,} chars"
    return f"  [{name}]{args_part}— called"


def main():
    print(f"Research Agent v{APP_VERSION} (type 'exit' to quit)")
    print(f"Model: {settings.model_name}")
    print("-" * 40)
    logger.info("Starting Research Agent v%s [%s]", APP_VERSION, settings.model_name)

    client = create_client(settings)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    session_tokens = {"input": 0, "output": 0, "total": 0}

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            logger.info("User exited session")
            print("Goodbye!")
            break

        logger.info("User: %s", user_input)

        try:
            # Use the streaming variant so we can show tool calls as they happen
            for event in run_agent_turn_streaming(user_input, messages, settings, client):
                if event["type"] == "tool_result":
                    print(_format_tool_status(event["name"], event["args"], event["result"]))
                elif event["type"] == "message":
                    print(f"\nAgent: {event['content']}")
                elif event["type"] == "tokens":
                    # The agent yields cumulative totals per turn; update session
                    session_tokens.update(event["data"])

        except (APIError, APIConnectionError, RateLimitError, AuthenticationError) as e:
            print(f"\nAgent: OpenAI API error — {e}")
            logger.error("OpenAI API error: %s", e)
        except Exception as e:
            print(f"\nAgent: Unexpected error — {e}")
            logger.exception("Unhandled error during agent turn")

        logger.info(
            "Session totals — input: %d, output: %d, total: %d",
            session_tokens["input"],
            session_tokens["output"],
            session_tokens["total"],
        )


if __name__ == "__main__":
    main()
