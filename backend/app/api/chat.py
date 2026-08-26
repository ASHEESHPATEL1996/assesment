import json
import time
import uuid

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, field_validator

from app.agents.crew import run_agentic_rag
from app.config import get_settings
from app.memory.store import save_message

router = APIRouter()


def _content_to_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(part for part in parts if part)
    return str(value)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str = "user"
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value) -> str:
        return _content_to_str(value)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    user: str | None = None
    temperature: float | None = None


def _session_id(request: Request, body: ChatCompletionRequest, session_header: str | None) -> str:
    return session_header or body.user or request.headers.get("x-openwebui-user-id") or "default"


def _question_and_history(messages: list[ChatMessage]) -> tuple[str, str]:
    question = ""
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            question = message.content.strip()
            break
    prior = messages[:-1] if messages and messages[-1].role == "user" else messages
    lines = []
    for message in prior[-get_settings().memory_max_turns :]:
        if message.role in {"user", "assistant"}:
            lines.append(f"{message.role}: {message.content}")
    return question, "\n".join(lines)


def _completion_payload(content: str) -> dict:
    settings = get_settings()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": settings.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream_payload(content: str) -> dict:
    settings = get_settings()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": settings.model_id,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ],
    }


@router.get("/v1/models")
@router.get("/models")
def list_models() -> dict:
    settings = get_settings()
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    question, history = _question_and_history(body.messages)
    if not question:
        raise HTTPException(status_code=400, detail="No user message provided")
    session_id = _session_id(request, body, x_session_id)
    save_message(session_id, "user", question)
    answer, _hits = run_agentic_rag(question, history)
    save_message(session_id, "assistant", answer)

    if body.stream:
        chunk = _stream_payload(answer)
        done = {
            **chunk,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

        def events():
            yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return _completion_payload(answer)
