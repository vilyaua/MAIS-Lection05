"""FastAPI web interface with Server-Sent Events (SSE) streaming.

Run with: uvicorn app:app --reload
Endpoints:
  GET /           — chat UI (single-page HTML)
  GET /api/info   — version + model metadata
  GET /api/chat?q — SSE stream of agent responses
  GET /api/reports — list saved reports
  GET /api/reports/{filename} — read a report
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

from agent import create_client, run_agent_turn_streaming
from config import APP_VERSION, SYSTEM_PROMPT, Settings

Path("logs").mkdir(exist_ok=True)

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
app = FastAPI(title="Research Agent", version=APP_VERSION)

# Conversation history for the web session (persists across requests within process)
web_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
web_client = create_client(settings)
session_tokens = {"input": 0, "output": 0, "total": 0}


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


def _format_tool_event(name: str, args: dict, result: str) -> dict:
    """Convert a tool result into a compact dict for the SSE stream."""
    primary_arg = _get_tool_call_args(name, args)
    if name == "knowledge_search":
        count = (
            result.count("---") + 1 if "---" in result else (0 if "No relevant" in result else 1)
        )
        return {"tool": name, "args": primary_arg, "detail": f"{count} documents found"}
    if name == "web_search":
        count = result.count("Title:")
        return {"tool": name, "args": primary_arg, "detail": f"{count} results found"}
    if name == "read_url":
        if result.startswith("Error"):
            return {"tool": name, "args": primary_arg, "detail": result[:80]}
        return {"tool": name, "args": primary_arg, "detail": f"extracted {len(result):,} chars"}
    if name == "write_report":
        return {"tool": name, "args": primary_arg, "detail": result}
    if name == "list_reports":
        count = result.count(". ")
        return {"tool": name, "args": "", "detail": f"{count} reports found"}
    if name == "read_file":
        if result.startswith("Error"):
            return {"tool": name, "args": primary_arg, "detail": result[:80]}
        return {"tool": name, "args": primary_arg, "detail": f"{len(result):,} chars"}
    return {"tool": name, "args": primary_arg, "detail": "called"}


def _sync_stream(prompt: str):
    """Run the ReAct loop in a sync context (called from a thread via run_in_executor)."""
    yield from run_agent_turn_streaming(prompt, web_messages, settings, web_client)


async def _stream_response(prompt: str) -> AsyncGenerator[str, None]:
    """Bridge sync ReAct streaming into async SSE.

    run_agent_turn_streaming() is synchronous and blocks the calling thread.
    To avoid blocking FastAPI's event loop, we run it in a thread pool and
    shuttle events through an asyncio.Queue back to the SSE generator.
    """
    global session_tokens

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    async def _produce():
        def _run():
            try:
                for event in _sync_stream(prompt):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except (APIError, APIConnectionError, RateLimitError, AuthenticationError) as e:
                err = f"OpenAI API error — {e}"
                logger.error("OpenAI API error: %s", e)
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "content": err})
            except Exception as e:
                err = f"Unexpected error — {e}"
                logger.exception("Unhandled error during agent turn")
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "content": err})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        await loop.run_in_executor(None, _run)

    task = asyncio.create_task(_produce())

    while True:
        event = await queue.get()
        if event is None:
            break

        if event["type"] == "error":
            yield f"data: {json.dumps({'type': 'message', 'content': event['content']})}\n\n"
            continue

        if event["type"] == "message":
            yield f"data: {json.dumps({'type': 'message', 'content': event['content']})}\n\n"

        elif event["type"] == "tool_result":
            tool_event = _format_tool_event(event["name"], event["args"], event["result"])
            yield f"data: {json.dumps({'type': 'tool', **tool_event})}\n\n"

        elif event["type"] == "tokens":
            session_tokens = event["data"]
            yield f"data: {json.dumps({'type': 'tokens', 'data': session_tokens})}\n\n"

        elif event["type"] == "done":
            pass  # will be sent after loop

    await task
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index():
    return CHAT_HTML


@app.get("/api/info")
async def info():
    return {
        "version": APP_VERSION,
        "model": settings.model_name,
        "tokens": session_tokens,
    }


@app.post("/api/reset")
async def reset():
    """Reset session: clear conversation history and token counters."""
    global session_tokens
    web_messages.clear()
    web_messages.append({"role": "system", "content": SYSTEM_PROMPT})
    session_tokens = {"input": 0, "output": 0, "total": 0}
    logger.info("Session reset by user")
    return {"status": "ok"}


@app.get("/api/chat")
async def chat(q: str):
    logger.info("User: %s", q)
    return StreamingResponse(
        _stream_response(q),
        media_type="text/event-stream",
    )


@app.get("/api/reports")
async def reports():
    """List all .md reports in output/ (newest first)."""
    output = Path(settings.output_dir)
    if not output.exists():
        return []
    files = sorted(
        (f for f in output.glob("*.md") if f.name[:1].isdigit()),
        key=lambda f: f.name,
        reverse=True,
    )
    return [{"name": f.name, "size": f.stat().st_size} for f in files]


@app.get("/api/reports/{filename}")
async def report_content(filename: str):
    """Return the raw markdown content of a report."""
    filepath = Path(settings.output_dir) / filename
    if not filepath.resolve().is_relative_to(Path(settings.output_dir).resolve()):
        raise HTTPException(status_code=403, detail="Access denied")
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return PlainTextResponse(filepath.read_text(encoding="utf-8"))


CHAT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f7f7f8; color: #1a1a1a; display: flex; height: 100vh; }
  .sidebar { width: 260px; background: #1e1e2e; color: #cdd6f4; padding: 20px;
             display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
  .sidebar h2 { font-size: 16px; color: #89b4fa; }
  .sidebar h3 { font-size: 13px; color: #89b4fa; margin-top: 4px; }
  .sidebar .meta { font-size: 12px; color: #6c7086; }
  .sidebar .metric { background: #313244; border-radius: 8px; padding: 10px; }
  .sidebar .metric .label { font-size: 11px; color: #6c7086; text-transform: uppercase; }
  .sidebar .metric .value { font-size: 20px; font-weight: 600; color: #cdd6f4; }
  .reports-list { display: flex; flex-direction: column; gap: 4px; }
  .report-item { font-size: 11px; color: #a6adc8; background: #313244; border-radius: 6px;
                 padding: 6px 8px; cursor: pointer; word-break: break-all; text-decoration: none; }
  .report-item:hover { background: #45475a; color: #cdd6f4; }
  .main { flex: 1; display: flex; flex-direction: column; }
  .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex;
              flex-direction: column; gap: 12px; }
  .msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; }
  .msg.user { align-self: flex-end; background: #2563eb; color: white; }
  .msg.assistant { align-self: flex-start; background: white; border: 1px solid #e5e7eb; }
  .msg.assistant pre { background: #f3f4f6; padding: 8px; border-radius: 6px;
                       overflow-x: auto; font-size: 13px; margin: 8px 0; }
  .msg.assistant code { font-size: 13px; }
  .tool-log { font-size: 12px; padding: 4px 16px; }
  .tool-log.knowledge_search { color: #ea580c; }
  .tool-log.web_search { color: #2563eb; }
  .tool-log.read_url { color: #7c3aed; }
  .tool-log.write_report { color: #059669; }
  .tool-log.list_reports { color: #d97706; }
  .tool-log.read_file { color: #0891b2; }
  .input-bar { padding: 16px 20px; background: white; border-top: 1px solid #e5e7eb;
               display: flex; gap: 8px; }
  .input-bar input { flex: 1; padding: 10px 14px; border: 1px solid #d1d5db;
                     border-radius: 8px; font-size: 14px; outline: none; }
  .input-bar input:focus { border-color: #2563eb; }
  .input-bar button { padding: 10px 20px; background: #2563eb; color: white;
                      border: none; border-radius: 8px; cursor: pointer; font-size: 14px; }
  .input-bar button:disabled { background: #93c5fd; cursor: not-allowed; }
  .btn-reset { width: 100%; padding: 8px; background: #45475a; color: #cdd6f4; border: none;
               border-radius: 8px; cursor: pointer; font-size: 13px; }
  .btn-reset:hover { background: #585b70; }
</style>
</head>
<body>
<div class="sidebar">
  <h2>Research Agent</h2>
  <button class="btn-reset" onclick="resetSession()">New Session</button>
  <div class="meta" id="meta">Loading...</div>
  <div class="metric"><div class="label">Input tokens</div><div class="value" id="t-in">0</div></div>
  <div class="metric"><div class="label">Output tokens</div><div class="value" id="t-out">0</div></div>
  <div class="metric"><div class="label">Total tokens</div><div class="value" id="t-total">0</div></div>
  <h3>Reports</h3>
  <div class="reports-list" id="reports">Loading...</div>
</div>
<div class="main">
  <div class="messages" id="messages"></div>
  <div class="input-bar">
    <input type="text" id="input" placeholder="Ask a research question..." autofocus />
    <button id="send" onclick="send()">Send</button>
  </div>
</div>
<script>
  const msgs = document.getElementById('messages');
  const input = document.getElementById('input');
  const btn = document.getElementById('send');

  fetch('/api/info').then(r=>r.json()).then(d=>{
    document.getElementById('meta').innerHTML =
      `v${d.version}<br><b>Model:</b> ${d.model}`;
    updateTokens(d.tokens);
  });

  function loadReports() {
    fetch('/api/reports').then(r=>r.json()).then(files=>{
      const el = document.getElementById('reports');
      if (!files.length) { el.textContent = 'No reports yet'; return; }
      el.innerHTML = files.map(f =>
        `<a class="report-item" href="/api/reports/${encodeURIComponent(f.name)}" target="_blank">${f.name}</a>`
      ).join('');
    });
  }
  loadReports();

  input.addEventListener('keydown', e => { if(e.key==='Enter' && !btn.disabled) send(); });

  function resetSession() {
    fetch('/api/reset', {method:'POST'}).then(r=>r.json()).then(()=>{
      msgs.innerHTML = '';
      updateTokens({input:0, output:0, total:0});
      loadReports();
      input.focus();
    });
  }

  function addMsg(role, html) {
    const d = document.createElement('div');
    d.className = 'msg ' + role;
    d.innerHTML = html;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
    return d;
  }

  function addTool(text, toolName) {
    const d = document.createElement('div');
    d.className = 'tool-log ' + (toolName || '');
    d.textContent = text;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function updateTokens(t) {
    document.getElementById('t-in').textContent = t.input.toLocaleString();
    document.getElementById('t-out').textContent = t.output.toLocaleString();
    document.getElementById('t-total').textContent = t.total.toLocaleString();
  }

  function formatMd(text) {
    return text
      .replace(/```(\\w*)\\n([\\s\\S]*?)```/g, '<pre><code>$2</code></pre>')
      .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
      .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>')
      .replace(/^### (.+)$/gm, '<h4>$1</h4>')
      .replace(/^## (.+)$/gm, '<h3>$1</h3>')
      .replace(/^# (.+)$/gm, '<h2>$1</h2>')
      .replace(/^- (.+)$/gm, '&bull; $1<br>')
      .replace(/\\n/g, '<br>');
  }

  async function send() {
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    btn.disabled = true;
    addMsg('user', q);
    const el = addMsg('assistant', '<em>Researching...</em>');

    const es = new EventSource('/api/chat?q=' + encodeURIComponent(q));
    let lastContent = '';

    es.onmessage = e => {
      const d = JSON.parse(e.data);
      if (d.type === 'message') { lastContent = d.content; el.innerHTML = formatMd(d.content); }
      if (d.type === 'tokens') updateTokens(d.data);
      if (d.type === 'tool') addTool(`\\u2192 ${d.tool}${d.args ? '("'+d.args+'")' : ''} \\u2014 ${d.detail}`, d.tool);
      if (d.type === 'done') { es.close(); btn.disabled = false; input.focus(); loadReports(); }
      msgs.scrollTop = msgs.scrollHeight;
    };
    es.onerror = () => { es.close(); btn.disabled = false; };
  }
</script>
</body>
</html>
"""
