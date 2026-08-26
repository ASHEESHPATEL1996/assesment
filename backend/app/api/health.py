from fastapi import APIRouter

from app.db import connect

router = APIRouter()


@router.get("/health")
def health() -> dict:
    postgres = "ok"
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception:
        postgres = "error"
    return {"status": "ok" if postgres == "ok" else "degraded", "postgres": postgres}
