from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore

from app.config import get_settings

_vector_store = None
_embed_model = None
_llm = None
_index = None


def get_embed_model() -> OllamaEmbedding:
    global _embed_model
    if _embed_model is None:
        settings = get_settings()
        _embed_model = OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
        Settings.embed_model = _embed_model
    return _embed_model


def get_llm() -> Ollama:
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = Ollama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            request_timeout=180.0,
            temperature=0.1,
            context_window=2048,
        )
        Settings.llm = _llm
    return _llm


def get_vector_store() -> PGVectorStore:
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        pg = settings.postgres
        params = {
            "database": pg["database"],
            "host": pg["host"],
            "password": pg["password"],
            "port": pg["port"],
            "user": pg["user"],
            "table_name": settings.vector_table_name,
            "embed_dim": settings.embed_dim,
        }
        try:
            _vector_store = PGVectorStore.from_params(**params, hybrid_search=False)
        except TypeError:
            _vector_store = PGVectorStore.from_params(**params)
    return _vector_store


def get_index() -> VectorStoreIndex:
    global _index
    if _index is None:
        storage = StorageContext.from_defaults(vector_store=get_vector_store())
        _index = VectorStoreIndex.from_vector_store(
            get_vector_store(),
            storage_context=storage,
            embed_model=get_embed_model(),
        )
    return _index


def reset_index_cache() -> None:
    global _index
    _index = None
