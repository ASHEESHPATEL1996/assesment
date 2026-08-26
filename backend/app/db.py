import logging

import psycopg2

from app.config import get_settings

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages (session_id, created_at);
"""


def connect():
    settings = get_settings()
    return psycopg2.connect(settings.database_url)


def init_db() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
        conn.commit()
    log.info("Database tables ready")
