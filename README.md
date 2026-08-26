# Agentic RAG

Local RAG chatbot. **OpenWebUI** is the frontend. A **FastAPI** backend answers from ingested documents using **Docling**, **LlamaIndex + PGVector**, **CrewAI**, **Ollama**, **Arize Phoenix**, and **RAGAS**.

The demo corpus is `data/sample/acme_handbook.md` (Acme Robotics employee handbook).

## How to run

Anyone with **Git** and **Docker** can run the full stack locally. You do not install Python, Ollama, or Postgres on the host.

### Prerequisites

1. [Git](https://git-scm.com/downloads)
2. [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows, macOS) or Docker Engine + Compose (Linux)
3. About **16 GB RAM** and **20 GB** free disk (model pull + images)

Start Docker and wait until it is fully running (`docker info` works).

### Clone and start

HTTPS:

```bash
git clone https://github.com/ASHEESHPATEL1996/assesment.git
cd assesment
cp .env.example .env
docker compose up --build
```

SSH:

```bash
git clone git@github.com:ASHEESHPATEL1996/assesment.git
cd assesment
cp .env.example .env
docker compose up --build
```

On Windows PowerShell you can use `copy .env.example .env` instead of `cp`.

The first start can take several minutes:

1. Images build (backend + branded OpenWebUI)
2. Ollama pulls `llama3.2:3b` and `nomic-embed-text`
3. Backend creates Postgres tables, seeds Phoenix prompts, and ingests the sample handbook

Leave the compose terminal open. In another terminal, from the same repo folder, use `docker compose ps` and the health checks below.

Stop with `Ctrl+C` in the compose terminal, or:

```bash
docker compose down
```

To start again later (skip clone):

```bash
cd assesment
docker compose up --build
```

### URLs

| Service | URL |
| --- | --- |
| OpenWebUI (chat) | http://localhost:3000 |
| FastAPI health | http://localhost:8000/health |
| FastAPI docs list | http://localhost:8000/documents |
| Phoenix traces | http://localhost:6006 |
| Ollama | http://localhost:11434 |
| Postgres | localhost:5432 (`rag` / `rag` / `ragdb`) |

### Use the chat

1. Open http://localhost:3000
2. Confirm the model **agentic-rag** is selected (that is this app, not a raw Ollama model)
3. Ask: `How many PTO days do employees get?`

A grounded answer should mention **20 days** and a **Sources** footer (`acme_handbook.md`). The first reply on CPU can take 1–3 minutes.

If `agentic-rag` is missing: **Admin Settings → Connections → OpenAI**, base URL `http://backend:8000/v1`, API key `unused`.

### Check that the stack is healthy

```bash
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/documents
curl http://localhost:8000/v1/models
```

Expect:

- `postgres`, `ollama`, `backend`, `open-webui` **healthy**
- health: `{"status":"ok","postgres":"ok"}`
- documents: `acme_handbook.md`
- models: `agentic-rag`

### Upload another file

```bash
curl -F "file=@data/sample/acme_handbook.md" http://localhost:8000/documents
```

Accepted types: `pdf`, `docx`, `md`, `txt`. Ingest can take a few minutes (each chunk is contextualized with the LLM).

### RAGAS evaluation

After the handbook is ingested:

```bash
docker compose exec backend python -m eval.run_ragas
```

Scores print in the terminal. Traces show up in Phoenix.

## Models

OpenWebUI shows **agentic-rag**. That is the FastAPI model id (`MODEL_ID`). Under the hood the stack uses three models:

| Role | Model | Where it runs | What it does |
| --- | --- | --- | --- |
| LLM | `llama3.2:3b` | Ollama | Contextualize chunks, CrewAI retriever + answer writer |
| Embeddings | `nomic-embed-text` (768-d) | Ollama | Embed chunks and queries for PGVector search |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Backend (sentence-transformers) | Rerank the top 12 vector hits down to 5 |

Ollama context length is capped at **2048** tokens (`OLLAMA_CONTEXT_LENGTH`) so the 3B model fits in ~16 GB RAM.

## Architecture

```
Browser
   │
   ▼
OpenWebUI :3000
   │  OpenAI-compatible /v1/chat/completions
   ▼
FastAPI backend :8000
   │
   ├─ ingest  → Docling → chunk → llama3.2 contextualize → nomic-embed-text → PGVector
   ├─ chat    → CrewAI retriever (search_docs) → rerank → CrewAI answer writer → citations
   ├─ memory  → Postgres chat_messages
   └─ traces  → Phoenix :6006
         │
         ├─ Ollama :11434   (llama3.2:3b, nomic-embed-text)
         └─ Postgres :5432  (documents, chat_messages, document_chunks)
```

### Compose services

| Service | Image / build | Role |
| --- | --- | --- |
| `postgres` | `pgvector/pgvector:pg16` | Relational store + vector index |
| `ollama` | `ollama/ollama` | Local LLM and embedding server |
| `ollama-pull` | one-shot | Pulls `llama3.2:3b` and `nomic-embed-text` |
| `phoenix` | `arizephoenix/phoenix` | Prompt store + traces |
| `backend` | `backend/Dockerfile` | FastAPI RAG API |
| `open-webui` | `openwebui/Dockerfile` | Chat UI pointed at the backend |

### Chat path

1. User message in OpenWebUI
2. `POST /v1/chat/completions` on FastAPI
3. Last 8 turns of history are passed in
4. CrewAI **Retrieval Specialist** must call `search_docs`
5. `search_docs` retrieves 12 nearest chunks from PGVector, reranks to 5
6. CrewAI **Answer Writer** writes from that evidence only
7. Response gets a **Sources** footer (`filename`, page, chunk id)
8. User + assistant messages are stored in `chat_messages`

If CrewAI kickoff fails, the API falls back to the retrieved passages so the user still gets evidence.

### Ingest path

1. Startup (if `INGEST_SAMPLES_ON_START=true`) or `POST /documents`
2. **Docling** extracts text (page-aware when possible)
3. **TokenTextSplitter** chunks at 512 tokens with 64 overlap
4. **llama3.2:3b** writes a 1–2 sentence prefix for each chunk (Anthropic-style contextual retrieval)
5. Prefixed text is embedded with **nomic-embed-text** and stored in PGVector table `document_chunks`
6. Filename is recorded in `documents`

### Observability

Phoenix keeps three prompts and receives traces:

- `contextualize-chunk`
- `retriever-agent`
- `answer-agent`

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | API + Postgres check |
| `GET` | `/v1/models` | Lists `agentic-rag` |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (OpenWebUI) |
| `POST` | `/documents` | Upload `pdf` / `docx` / `md` / `txt` |
| `GET` | `/documents` | List ingested files |

Example chat call:

```bash
curl http://localhost:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"agentic-rag\",\"messages\":[{\"role\":\"user\",\"content\":\"How many PTO days do employees get?\"}]}"
```

## Configuration

Copy `.env.example` to `.env`. Docker Compose also sets the same keys on the backend container.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_MODEL` | `llama3.2:3b` | Chat / contextualize model on Ollama |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model on Ollama |
| `EMBED_DIM` | `768` | Must match the embedding model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Chunking |
| `SIMILARITY_TOP_K` | `12` | Vector neighbors |
| `RERANK_TOP_N` | `5` | Passages sent to the writer |
| `MEMORY_MAX_TURNS` | `8` | Chat history window |
| `MODEL_ID` | `agentic-rag` | Name OpenWebUI sees |
| `INGEST_SAMPLES_ON_START` | `true` | Ingest `data/sample` on boot |

To change the LLM or embeddings, update `LLM_MODEL` / `EMBEDDING_MODEL`, pull the new Ollama tags, and restart. If you change embedding dimension, wipe the `postgres_data` volume so PGVector is recreated.

## Layout

```
backend/app/          FastAPI, RAG, CrewAI, Phoenix
backend/eval/         RAGAS dataset + runner
openwebui/            OpenWebUI image + custom.css
data/sample/          Demo handbook
docker-compose.yml
.env.example
```

## Troubleshooting

**Docker daemon not running**  
Start Docker Desktop, wait until it is ready, then `docker compose up --build` again.

**Port 3000 or 8000 already allocated**  
Another stack (for example `assignment-*`) is using the port. Stop those containers, then start this project.

**`Sample ingest skipped` / empty `/documents`**  
Watch backend logs. Ingest needs Ollama up and can take a few minutes. Retry with the `curl -F` upload command.

**Chat times out or Crew falls back to raw passages**  
Normal on CPU for `llama3.2:3b`. Wait and retry. Evidence should still include handbook text.

**OpenWebUI cannot see `agentic-rag`**  
Set OpenAI base URL to `http://backend:8000/v1` (from inside Docker) with API key `unused`. Do not point OpenWebUI at Ollama directly; Ollama is only used by the backend.
