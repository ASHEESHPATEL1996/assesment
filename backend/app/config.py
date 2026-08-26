from functools import lru_cache
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    database_url: str = "postgresql://rag:rag@localhost:5432/ragdb"
    ollama_base_url: str = "http://localhost:11434"
    phoenix_collector_endpoint: str = "http://localhost:6006"
    phoenix_base_url: str = "http://localhost:6006"

    llm_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"
    embed_dim: int = 768
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 512
    chunk_overlap: int = 64
    similarity_top_k: int = 12
    rerank_top_n: int = 5
    memory_max_turns: int = 8

    model_id: str = "agentic-rag"
    ingest_samples_on_start: bool = True
    openai_api_key: str = "ollama"
    vector_table_name: str = "document_chunks"
    sample_dir: str = "/app/data/sample"

    @property
    def postgres(self) -> dict:
        parsed = urlparse(self.database_url)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "user": parsed.username or "rag",
            "password": parsed.password or "rag",
            "database": (parsed.path or "/ragdb").lstrip("/"),
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
