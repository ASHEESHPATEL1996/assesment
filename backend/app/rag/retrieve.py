import logging

from app.config import get_settings
from app.rag.vectorstore import get_index

log = logging.getLogger(__name__)

_reranker_model = None
_last_hits: list[dict] = []


def get_last_hits() -> list[dict]:
    return list(_last_hits)


def set_last_hits(hits: list[dict]) -> None:
    global _last_hits
    _last_hits = list(hits)


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        _reranker_model = CrossEncoder(settings.rerank_model)
    return _reranker_model


def retrieve(query: str) -> list[dict]:
    settings = get_settings()
    try:
        nodes = get_index().as_retriever(similarity_top_k=settings.similarity_top_k).retrieve(query)
    except Exception as exc:
        log.warning("Vector retrieve failed: %s", exc)
        return []
    if not nodes:
        return []

    pairs = [(query, node.get_content()) for node in nodes]
    try:
        scores = _get_reranker().predict(pairs)
        ranked = sorted(zip(nodes, scores), key=lambda item: float(item[1]), reverse=True)
    except Exception as exc:
        log.warning("Rerank failed, using vector order: %s", exc)
        ranked = [(node, getattr(node, "score", 0.0) or 0.0) for node in nodes]

    hits = []
    for node, score in ranked[: settings.rerank_top_n]:
        metadata = node.metadata or {}
        hits.append(
            {
                "text": node.get_content(),
                "filename": metadata.get("filename", "unknown"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "score": float(score),
            }
        )
    set_last_hits(hits)
    return hits
