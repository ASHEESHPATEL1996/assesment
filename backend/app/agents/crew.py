import logging
import os

from pydantic import BaseModel, Field

from app.config import get_settings
from app.observability.phoenix import get_prompt
from app.rag.citations import append_citations, hits_to_tool_text
from app.rag.retrieve import get_last_hits, retrieve, set_last_hits

log = logging.getLogger(__name__)


class SearchInput(BaseModel):
    query: str = Field(..., description="Natural language search query for the knowledge base")


def _search_docs(query: str) -> str:
    hits = retrieve(query)
    return hits_to_tool_text(hits)


def _import_crew():
    try:
        from crewai import LLM, Agent, Crew, Process, Task
    except ImportError:
        from crewai import Agent, Crew, Process, Task
        from crewai.llm import LLM
    try:
        from crewai.tools import BaseTool
    except ImportError:
        from crewai_tools import BaseTool
    return LLM, Agent, Crew, Process, Task, BaseTool


def _build_llm(LLM, settings):
    kwargs = {
        "model": f"ollama/{settings.llm_model}",
        "api_key": settings.openai_api_key,
        "temperature": 0.1,
    }
    try:
        return LLM(**kwargs, base_url=settings.ollama_base_url)
    except TypeError:
        return LLM(**kwargs, api_base=settings.ollama_base_url)


def _build_agent(Agent, **kwargs):
    attempts = [
        {**kwargs, "memory": False},
        kwargs,
        {key: value for key, value in kwargs.items() if key not in {"memory", "max_iter"}},
    ]
    last_error = None
    for attempt in attempts:
        try:
            return Agent(**attempt)
        except TypeError as exc:
            last_error = exc
    raise last_error


def run_agentic_rag(question: str, history: str = "") -> tuple[str, list[dict]]:
    LLM, Agent, Crew, Process, Task, BaseTool = _import_crew()

    class SearchDocsTool(BaseTool):
        name: str = "search_docs"
        description: str = (
            "Search ingested documents and return the top reranked passages "
            "with filename, page, and chunk metadata."
        )
        args_schema: type[BaseModel] = SearchInput

        def _run(self, query: str) -> str:
            return _search_docs(query)

    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    os.environ.setdefault("OLLAMA_API_BASE", settings.ollama_base_url)
    set_last_hits([])

    llm = _build_llm(LLM, settings)
    search_tool = SearchDocsTool()
    retriever_prompt = get_prompt("retriever-agent")
    answer_prompt = get_prompt("answer-agent")

    retriever = _build_agent(
        Agent,
        role="Retrieval Specialist",
        goal="Find the most relevant document passages for the user's question.",
        backstory=retriever_prompt,
        tools=[search_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )
    writer = _build_agent(
        Agent,
        role="Answer Writer",
        goal="Write a grounded answer with citations from retrieved evidence.",
        backstory=answer_prompt,
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=2,
    )

    retrieve_task = Task(
        description=(
            f"Search the knowledge base for this question:\n{question}\n"
            "You must call the search_docs tool."
        ),
        expected_output="Relevant passages with filenames and page numbers.",
        agent=retriever,
    )
    answer_task = Task(
        description=(
            f"User question: {question}\n\n"
            f"Conversation history:\n{history or '(none)'}\n\n"
            "Using the retrieved evidence from the previous task, write a concise answer. "
            "Cite sources as [filename, p.N]. If evidence is insufficient, say you do not know."
        ),
        expected_output="A grounded answer followed by a Sources list.",
        agent=writer,
        context=[retrieve_task],
    )

    crew_kwargs = {
        "agents": [retriever, writer],
        "tasks": [retrieve_task, answer_task],
        "process": Process.sequential,
        "verbose": False,
    }
    try:
        crew = Crew(**crew_kwargs, memory=False)
    except TypeError:
        crew = Crew(**crew_kwargs)
    try:
        result = crew.kickoff()
        answer = str(result)
    except Exception as exc:
        log.warning("Crew kickoff failed, answering from direct retrieve: %s", exc)
        answer = ""

    hits = get_last_hits()
    if not hits:
        hits = retrieve(question)
    if not answer.strip():
        evidence = hits_to_tool_text(hits)
        answer = (
            f"I could not complete the agent run. Evidence for the question:\n{evidence}"
            if hits
            else "I do not know. No relevant documents were found."
        )
    return append_citations(answer, hits), hits
