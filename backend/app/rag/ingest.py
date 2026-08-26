import logging
import uuid
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TextNode

from app.config import get_settings
from app.db import connect
from app.observability.phoenix import get_prompt
from app.rag.vectorstore import get_embed_model, get_index, get_llm, reset_index_cache

log = logging.getLogger(__name__)


def _page_texts(path: Path) -> list[tuple[int, str]]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    pages: dict[int, list[str]] = {}
    used_items = False
    for item, _level in result.document.iterate_items():
        text = getattr(item, "text", None)
        if not text or not str(text).strip():
            continue
        used_items = True
        page_no = 1
        prov = getattr(item, "prov", None) or []
        if prov:
            page_no = getattr(prov[0], "page_no", None) or 1
        pages.setdefault(int(page_no), []).append(str(text).strip())
    if used_items and pages:
        return [(page, "\n".join(chunks)) for page, chunks in sorted(pages.items())]
    markdown = result.document.export_to_markdown()
    return [(1, markdown)]


def _contextualize(filename: str, chunk: str) -> str:
    prompt = get_prompt("contextualize-chunk", {"filename": filename, "chunk": chunk[:1500]})
    try:
        prefix = str(get_llm().complete(prompt)).strip()
    except Exception as exc:
        log.warning("Contextualize failed, using filename prefix: %s", exc)
        prefix = f"This chunk is from {filename}."
    if not prefix:
        prefix = f"This chunk is from {filename}."
    return f"{prefix}\n\n{chunk}"


def _record_document(doc_id: str, filename: str) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (id, filename) VALUES (%s, %s)",
                (doc_id, filename),
            )
        conn.commit()


def list_documents() -> list[dict]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, filename, created_at FROM documents ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
    return [
        {"id": doc_id, "filename": filename, "created_at": created_at.isoformat()}
        for doc_id, filename, created_at in rows
    ]


def ingest_file(path: str | Path, filename: str | None = None) -> dict:
    settings = get_settings()
    path = Path(path)
    filename = filename or path.name
    doc_id = str(uuid.uuid4())
    splitter = TokenTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    documents: list[Document] = []
    chunk_id = 0
    for page, text in _page_texts(path):
        if not text.strip():
            continue
        page_doc = Document(
            text=text,
            metadata={"filename": filename, "page": page, "doc_id": doc_id},
        )
        for node in splitter.get_nodes_from_documents([page_doc]):
            chunk_id += 1
            raw = node.get_content().strip()
            if not raw:
                continue
            contextual = _contextualize(filename, raw)
            documents.append(
                Document(
                    text=contextual,
                    doc_id=doc_id,
                    metadata={
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": page,
                        "chunk_id": chunk_id,
                    },
                )
            )

    if not documents:
        raise ValueError(f"No text extracted from {filename}")

    get_embed_model()
    nodes = [
        TextNode(
            text=doc.text,
            id_=f"{doc_id}-{doc.metadata['chunk_id']}",
            metadata=doc.metadata,
        )
        for doc in documents
    ]
    get_index().insert_nodes(nodes)
    _record_document(doc_id, filename)
    reset_index_cache()
    log.info("Ingested %s (%s chunks)", filename, len(nodes))
    return {"id": doc_id, "filename": filename, "chunks": len(nodes)}


def ingest_samples() -> list[dict]:
    settings = get_settings()
    candidates = [
        Path(settings.sample_dir),
        Path("/app/data/sample"),
        Path(__file__).resolve().parents[3] / "data" / "sample",
        Path(__file__).resolve().parents[2].parent / "data" / "sample",
    ]
    sample_dir = next((path for path in candidates if path.exists()), None)
    if sample_dir is None:
        log.warning("Sample directory missing. Looked at: %s", candidates)
        return []
    if list_documents():
        log.info("Documents already present; skipping sample ingest")
        return []
    results = []
    for path in sorted(sample_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx", ".md", ".txt"}:
            continue
        results.append(ingest_file(path, path.name))
    return results
