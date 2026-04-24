# 🔍 AI Search Engine — Backend

> Production-grade AI search backend. Real-time answers with citations, iterative deep research, file intelligence, and an extensible plugin system — powered entirely by **free-tier services**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          Client                               │
│  /search/instant │ /agent/run │ /rag/query │ /workspace/chat │
└────────────┬───────────┬────────────┬──────────────┬─────────┘
             │           │            │              │
┌────────────▼───────────▼────────────▼──────────────▼─────────┐
│                    FastAPI + Middleware                        │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────────────────┐ │
│  │ Rate Limit   │ │ Request ID   │ │ Global Exception       │ │
│  │ (Redis)      │ │ Tracing      │ │ Handler                │ │
│  └─────────────┘ └──────────────┘ └────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│               Agent & RAG Orchestration (LLM)                  │
│  Plan → Execute (Web Search, RAG Query, Tools) → Review → Synthesize   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Search Agent    │  │  Agent Planner   │  │  RAG System  │ │
│  │  DDG + Scrape    │  │  ReAct Loop      │  │  Qdrant      │ │
│  │  + LLM Synth     │  │  Tool Registry   │  │  RRF + Rerank│ │
│  │  + Citations     │  │  + Executor      │  │  SSE Streams │ │
│  └────────┬────────┘  └────────┬─────────┘  └──────┬──────┘ │
│           │                    │                    │         │
├───────────▼────────────────────▼────────────────────▼─────────┤
│  MongoDB (NoSQL)  │  Redis (Cache, RateLimits)  │   Qdrant (Vectors)  │
└───────────────────┴─────────────────────────────┴─────────────────────┘
```

## Tech Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| **Runtime** | Python 3.11 + FastAPI | Free |
| **LLM** | Groq (Llama 3 70B / 8B) | Free tier |
| **Search** | DuckDuckGo + Trafilatura | Free |
| **Vector DB** | Qdrant (Self-hosted via Docker) | Free |
| **Embeddings** | `sentence-transformers` (Local) | Free |
| **Reranker** | Cross-Encoder (Local) | Free |
| **Database** | MongoDB (Docker) | Free |
| **Cache/Queue** | Redis (Docker) | Free |

## Quick Start

```bash
# 1. Clone and enter backend
cd backend

# 2. Set up Python environment
python -m venv venv && source venv/bin/activate
pip install -e .

# 3. Configure
cp .env.example .env
# Edit .env → add your GROQ_API_KEY

# 4. Start infrastructure
docker compose up -d

# 5. Run the server
uvicorn app.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for interactive API documentation.

---

## API Reference

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (pings MongoDB + Redis) |

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/search/instant` | Real-time AI search with SSE streaming |

### Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/agent/tools` | List all available agent tools |
| POST | `/api/v1/agent/run` | Run autonomous agent (chain-of-thought) with SSE streaming |

### RAG & Workspace

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/rag/upload` | Upload PDF/CSV for Qdrant vector indexing |
| POST | `/api/v1/rag/query` | Hybrid RAG query (Vector + Keyword search run + RRF) |
| GET | `/api/v1/rag/workspace/{id}/documents` | List documents inside a workspace |
| DELETE | `/api/v1/rag/workspace/{id}` | Delete workspace and all its vectors |

### Legacy Workspace (Deprecating)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/workspace/upload` | Upload PDF/CSV for analysis |
| POST | `/api/v1/workspace/chat` | Chat with uploaded files (SSE) |


---

### `POST /api/v1/search/instant`

**Request:**
```json
{ "query": "Why did the stock market drop today?", "session_id": null }
```

**Response:** Server-Sent Events stream
```
data: {"type": "sources", "data": [{"id": 1, "title": "...", "url": "...", "snippet": "..."}]}
data: {"type": "chunk", "text": "The stock market experienced"}
data: {"type": "chunk", "text": " a significant decline today"}
data: {"type": "done"}
```

### `POST /api/v1/search/deep/`

**Request:**
```json
{ "query": "Create a competitive analysis of the EV market in Southeast Asia" }
```

**Response:**
```json
{ "job_id": "550e8400-e29b-41d4-a716-446655440000", "status": "pending" }
```

### `GET /api/v1/search/deep/{job_id}`

**Response (completed):**
```json
{
  "status": "completed",
  "report": "# EV Market Analysis\n\n## Key Players\n...",
  "sources_count": "12"
}
```

---

## Production Features

- **Structured JSON Logging** — Every request traced with `X-Request-ID`
- **Rate Limiting** — Redis-backed, 60 req/min/IP (fail-open)
- **Response Caching** — SHA256-keyed Redis cache with TTL
- **Retry with Backoff** — All LLM calls use tenacity (3 attempts, exponential)
- **Graceful Degradation** — No sources? Helpful message. Node fails? Fallback. Redis down? Allow through.
- **Per-Node Error Recovery** — Deep research graph continues even if individual nodes fail
- **Health Checks** — `/health` pings both MongoDB and Redis, returns `degraded` if either is down
- **Non-root Docker** — Multi-stage build with gunicorn workers

## Environment Variables

See [`.env.example`](.env.example) for all available configuration options.
