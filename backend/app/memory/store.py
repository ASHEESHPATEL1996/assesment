from app.db import connect


def save_message(session_id: str, role: str, content: str) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )
        conn.commit()


def get_history(session_id: str, limit: int = 16) -> list[dict]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM (
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) recent
                ORDER BY created_at ASC
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
    return [{"role": role, "content": content} for role, content in rows]
