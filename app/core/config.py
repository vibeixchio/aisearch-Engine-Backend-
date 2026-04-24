from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Search Engine"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]  # Restrict in production

    # ── PostgreSQL (Supabase-hosted) ──────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_search"
    # Supabase connection pooling settings
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DEFAULT_TTL: int = 600       # 10 minutes default
    REDIS_USER_TTL: int = 900          # 15 minutes for user profiles
    REDIS_WORKSPACE_TTL: int = 600     # 10 minutes for workspace data
    REDIS_SEARCH_TTL: int = 300        # 5 minutes for search results
    REDIS_RAG_TTL: int = 600           # 10 minutes for RAG results
    REDIS_SESSION_TTL: int = 900       # 15 minutes for session cache

    # ── Security ──────────────────────────────────────────────────────────────
    SECRET_KEY: str = ""  # MUST be set in .env — no fallback
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_MIN_LENGTH: int = 8
    LOGIN_MAX_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15

    # ── Email (Resend) ────────────────────────────────────────────────────────
    RESEND_API_KEY: str = ""
    EMAIL_FROM: str = "noreply@aisearch.in"
    EMAIL_VERIFICATION_EXPIRE_MINUTES: int = 30

    # ── LLM — Groq (free tier, blazing fast) ─────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_ROUTER_MODEL: str = "llama3-8b-8192"
    GROQ_SYNTHESIS_MODEL: str = "llama3-70b-8192"

    # ── LLM — Google Gemini (free fallback) ───────────────────────────────────
    GOOGLE_API_KEY: str = ""

    # ── Qdrant Vector DB ──────────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "documents"

    # ── Embedding & Reranker (local, no API key needed) ───────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── RAG tuning ────────────────────────────────────────────────────────────
    RAG_TOP_K: int = 6
    RAG_CHUNK_SIZE: int = 512
    RAG_CHUNK_OVERLAP: int = 64

    # ── Agent ─────────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 8

    # ── Rate limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_RPM: int = 60

    # ── Search ────────────────────────────────────────────────────────────────
    MAX_SEARCH_RESULTS: int = 5
    SCRAPE_TIMEOUT_SECONDS: float = 8.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
