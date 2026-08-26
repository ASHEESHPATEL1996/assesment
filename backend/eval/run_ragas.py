"""Run RAGAS evaluation against the local agentic RAG pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.observability.phoenix import init_tracing

init_tracing()

from datasets import Dataset
from ragas import evaluate

from app.agents.crew import run_agentic_rag
from app.config import get_settings

DATASET = Path(__file__).with_name("dataset.json")


def load_cases() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def collect_rows(cases: list[dict]) -> dict:
    questions, answers, contexts, ground_truths = [], [], [], []
    for case in cases:
        question = case["question"]
        answer, hits = run_agentic_rag(question, history="")
        questions.append(question)
        answers.append(answer)
        contexts.append([hit["text"] for hit in hits] or [""])
        ground_truths.append(case["ground_truth"])
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    }


def build_metrics():
    try:
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        return [faithfulness, answer_relevancy, context_precision]
    except ImportError:
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            ResponseRelevancy,
        )

        return [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()]


def wrap_llm_embeddings():
    settings = get_settings()
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        return llm, embeddings


def main() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    rows = collect_rows(load_cases())
    dataset = Dataset.from_dict(rows)
    metrics = build_metrics()
    llm, embeddings = wrap_llm_embeddings()
    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    print(result)
    out = Path(__file__).with_name("last_results.json")
    try:
        if hasattr(result, "to_pandas"):
            payload = result.to_pandas().to_dict(orient="records")
        else:
            payload = {"result": str(result)}
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {out}")
    except Exception as exc:
        print("Could not write results file:", exc)


if __name__ == "__main__":
    main()
