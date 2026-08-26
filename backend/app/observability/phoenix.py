import logging
import os

from app.config import get_settings

log = logging.getLogger(__name__)

PROMPT_CONTEXTUALIZE = (
    "You write a short situating context for a document chunk, Anthropic-style.\n"
    "Document filename: {{filename}}\n"
    "Chunk:\n{{chunk}}\n\n"
    "Return 1-2 sentences that locate this chunk in the document. "
    "No preamble, no quotes, no extra commentary."
)

PROMPT_RETRIEVER = (
    "You are a retrieval specialist for an internal knowledge base. "
    "Always call the search_docs tool with the user's question. "
    "Return only the evidence the tool gives you. Do not invent passages."
)

PROMPT_ANSWER = (
    "You are a careful answer writer for Acme Robotics internal docs. "
    "Use only retrieved evidence and the conversation history. "
    "If the evidence is missing, say you do not know. "
    "Cite sources as [filename, p.N]. Never invent citations. "
    "Keep answers concise."
)

DEFAULT_PROMPTS = {
    "contextualize-chunk": PROMPT_CONTEXTUALIZE,
    "retriever-agent": PROMPT_RETRIEVER,
    "answer-agent": PROMPT_ANSWER,
}

_phoenix_client = None


def init_tracing() -> None:
    settings = get_settings()
    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", settings.phoenix_collector_endpoint)
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    try:
        from phoenix.otel import register
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from openinference.instrumentation.crewai import CrewAIInstrumentor
        from openinference.instrumentation.ollama import OllamaInstrumentor

        tracer_provider = register(project_name="agentic-rag", auto_instrument=True)
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
        OllamaInstrumentor().instrument(tracer_provider=tracer_provider)
        log.info("Phoenix tracing registered")
    except Exception as exc:
        log.warning("Phoenix tracing not initialized: %s", exc)


def _client():
    global _phoenix_client
    if _phoenix_client is None:
        from phoenix.client import Client

        settings = get_settings()
        _phoenix_client = Client(base_url=settings.phoenix_base_url)
    return _phoenix_client


def _extract_prompt_text(prompt, variables: dict | None = None) -> str:
    variables = variables or {}
    try:
        formatted = prompt.format(variables=variables)
        if isinstance(formatted, dict):
            messages = formatted.get("messages") or []
            return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        if hasattr(formatted, "messages"):
            parts = []
            for message in formatted.messages:
                content = getattr(message, "content", None)
                parts.append(str(content if content is not None else message))
            return "\n".join(parts)
        return str(formatted)
    except Exception:
        template = getattr(prompt, "template", "") or ""
        text = str(template)
        for key, value in variables.items():
            text = text.replace("{{" + key + "}}", str(value))
        return text


def _fill(template: str, variables: dict | None = None) -> str:
    text = template
    for key, value in (variables or {}).items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def seed_prompts() -> None:
    try:
        from phoenix.client.types import PromptVersion
    except Exception as exc:
        log.warning("Phoenix prompt client unavailable: %s", exc)
        return

    settings = get_settings()
    client = _client()
    for name, content in DEFAULT_PROMPTS.items():
        try:
            client.prompts.get(prompt_identifier=name)
            log.info("Phoenix prompt already present: %s", name)
            continue
        except Exception:
            pass
        try:
            client.prompts.create(
                name=name,
                version=PromptVersion(
                    [{"role": "system", "content": content}],
                    model_name=settings.llm_model,
                ),
                prompt_description=f"Agentic RAG prompt: {name}",
            )
            log.info("Seeded Phoenix prompt: %s", name)
        except Exception as exc:
            log.warning("Could not seed Phoenix prompt %s: %s", name, exc)


def get_prompt(name: str, variables: dict | None = None) -> str:
    fallback = _fill(DEFAULT_PROMPTS.get(name, ""), variables)
    try:
        prompt = _client().prompts.get(prompt_identifier=name)
        text = _extract_prompt_text(prompt, variables)
        return text.strip() or fallback
    except Exception as exc:
        log.warning("Falling back to local prompt %s: %s", name, exc)
        return fallback
