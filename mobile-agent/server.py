import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

load_dotenv()

DEFAULT_WORKSPACE = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR") or DEFAULT_WORKSPACE).resolve()
API_KEY = os.environ.get("CURSOR_API_KEY", "")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")
PORT = int(os.environ.get("PORT", "8787"))


def require_config() -> None:
    missing = []
    if not API_KEY:
        missing.append("CURSOR_API_KEY")
    if not ACCESS_TOKEN:
        missing.append("ACCESS_TOKEN")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
    if not WORKSPACE.is_dir():
        raise RuntimeError(f"WORKSPACE_DIR does not exist: {WORKSPACE}")


def verify_token(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        token = request.query_params.get("token", "")
    if token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid access token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    require_config()
    from cursor_sdk import AsyncClient, LocalAgentOptions

    bridge = await AsyncClient.launch_bridge(workspace=str(WORKSPACE))
    async with bridge as client:
        agent_handle = await client.agents.create(
            model="composer-2.5",
            api_key=API_KEY,
            local=LocalAgentOptions(cwd=str(WORKSPACE)),
        )
        async with agent_handle as agent:
            app.state.agent = agent
            yield


app = FastAPI(title="Local Mobile Agent", lifespan=lifespan)

CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>Local Agent</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0f1115;
      --panel: #171a21;
      --text: #e8eaed;
      --muted: #9aa0a6;
      --accent: #6ea8fe;
      --user: #1f3b63;
      --assistant: #1b1f27;
      --border: #2a2f3a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100dvh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
      font-weight: 600;
    }
    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .bubble {
      max-width: 92%;
      padding: 12px 14px;
      border-radius: 14px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .user { align-self: flex-end; background: var(--user); }
    .assistant { align-self: flex-start; background: var(--assistant); border: 1px solid var(--border); }
    .meta { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    form {
      display: flex;
      gap: 8px;
      padding: 12px;
      border-top: 1px solid var(--border);
      background: var(--panel);
    }
    input, textarea, button {
      font: inherit;
    }
    #prompt {
      flex: 1;
      resize: none;
      min-height: 44px;
      max-height: 120px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #0b0d11;
      color: var(--text);
      padding: 10px 12px;
    }
    button {
      border: 0;
      border-radius: 12px;
      background: var(--accent);
      color: #081018;
      font-weight: 600;
      padding: 0 16px;
      min-width: 72px;
    }
    button:disabled { opacity: 0.6; }
    #gate {
      padding: 24px 16px;
      display: grid;
      gap: 12px;
      max-width: 420px;
      margin: auto;
    }
    #gate input {
      width: 100%;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #0b0d11;
      color: var(--text);
    }
    .hidden { display: none !important; }
  </style>
</head>
<body>
  <div id="gate">
    <h2>Local Agent</h2>
    <p>Enter your access token to connect to this computer.</p>
    <input id="token" type="password" placeholder="Access token" autocomplete="current-password" />
    <button id="connect">Connect</button>
  </div>

  <div id="app" class="hidden">
    <header>Local Agent</header>
    <div id="messages"></div>
    <form id="chat-form">
      <textarea id="prompt" rows="1" placeholder="Ask the agent to work on your files..."></textarea>
      <button type="submit">Send</button>
    </form>
  </div>

  <script>
    const gate = document.getElementById("gate");
    const app = document.getElementById("app");
    const tokenInput = document.getElementById("token");
    const connectBtn = document.getElementById("connect");
    const messages = document.getElementById("messages");
    const form = document.getElementById("chat-form");
    const prompt = document.getElementById("prompt");
    let token = localStorage.getItem("mobileAgentToken") || "";

    function addBubble(role, text) {
      const wrap = document.createElement("div");
      wrap.className = "bubble " + role;
      wrap.textContent = text;
      messages.appendChild(wrap);
      messages.scrollTop = messages.scrollHeight;
      return wrap;
    }

    function showApp() {
      gate.classList.add("hidden");
      app.classList.remove("hidden");
    }

    async function connect() {
      token = tokenInput.value.trim();
      if (!token) return;
      const res = await fetch("/api/health?token=" + encodeURIComponent(token));
      if (!res.ok) {
        alert("Invalid token");
        return;
      }
      localStorage.setItem("mobileAgentToken", token);
      showApp();
    }

    connectBtn.addEventListener("click", connect);
    tokenInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") connect();
    });

    if (token) {
      fetch("/api/health?token=" + encodeURIComponent(token))
        .then((res) => res.ok ? showApp() : localStorage.removeItem("mobileAgentToken"))
        .catch(() => localStorage.removeItem("mobileAgentToken"));
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = prompt.value.trim();
      if (!text) return;
      prompt.value = "";
      addBubble("user", text);
      const assistant = addBubble("assistant", "");
      const submitBtn = form.querySelector("button");
      submitBtn.disabled = true;

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
          },
          body: JSON.stringify({ message: text }),
        });
        if (!res.ok) throw new Error("Request failed");
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() || "";
          for (const part of parts) {
            if (!part.startsWith("data: ")) continue;
            const payload = JSON.parse(part.slice(6));
            if (payload.text) assistant.textContent += payload.text;
            if (payload.error) assistant.textContent += "\n\nError: " + payload.error;
            messages.scrollTop = messages.scrollHeight;
          }
        }
      } catch (err) {
        assistant.textContent += "\n\nFailed: " + err.message;
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return CHAT_HTML


@app.get("/api/health")
async def health(request: Request) -> dict:
    verify_token(request)
    return {"ok": True, "workspace": str(WORKSPACE)}


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    verify_token(request)
    body = await request.json()
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    agent = request.app.state.agent

    async def event_stream():
        try:
            run = await agent.send(message)
            async for chunk in run.iter_text():
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            result = await run.wait()
            yield f"data: {json.dumps({'done': True, 'status': result.status})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    require_config()
    uvicorn.run("server:app", host="127.0.0.1", port=PORT, reload=False)
