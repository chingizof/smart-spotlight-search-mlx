import json
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Resolve relative paths in imessages.py (lancedb/, chat-history/) from repo root
REPO_ROOT = Path(__file__).parent.parent
os.chdir(REPO_ROOT)

sys.path.insert(0, str(Path(__file__).parent))
from core import MODEL, agent_stream, ensure_model

app = FastAPI(title="Smart Spotlight Search Agent")
UI_HTML = (Path(__file__).parent / "ui.html").read_text()


class ChatRequest(BaseModel):
    query: str
    after: str | None = None
    before: str | None = None


@app.get("/")
def root():
    return HTMLResponse(UI_HTML)


@app.post("/chat")
def chat(req: ChatRequest):
    def event_stream():
        for event in agent_stream(req.query, req.after, req.before):
            yield f"data: {json.dumps(event)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


if __name__ == "__main__":
    print("=" * 50)
    print("  Smart Spotlight Search Agent")
    print("=" * 50)
    ensure_model()
    print("\n  http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
