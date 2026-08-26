import logging
import os
import threading
from contextlib import asynccontextmanager

from app.observability.phoenix import init_tracing

init_tracing()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.config import get_settings
from app.db import init_db
from app.observability.phoenix import seed_prompts
from app.rag.ingest import ingest_samples
from app.rag.vectorstore import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _safe_ingest_samples() -> None:
    try:
        ingest_samples()
    except Exception as exc:
        log.warning("Sample ingest skipped: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    os.environ.setdefault("OLLAMA_API_BASE", settings.ollama_base_url)
    init_db()
    get_vector_store()
    seed_prompts()
    if settings.ingest_samples_on_start:
        threading.Thread(target=_safe_ingest_samples, daemon=True).start()
    yield


app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ingest_router)
