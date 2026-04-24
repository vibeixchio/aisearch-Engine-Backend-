import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.middleware import RequestContextMiddleware, RateLimitMiddleware, global_exception_handler
from app.core.security_headers import SecurityHeadersMiddleware
from app.db.database import init_db, close_db
from app.db.redis_client import connect_to_redis, close_redis_connection, redis_client

# ---------------------------------------------------------------------------
# Bootstrap logging BEFORE anything else
# ---------------------------------------------------------------------------
setup_logging(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Search Engine backend v%s", settings.VERSION)

    # Initialize PostgreSQL (Supabase)
    await init_db()

    # Initialize Redis
    await connect_to_redis()

    # Initialize Qdrant collection (creates if not exists)
    try:
        from app.services.vector_store import ensure_collection
        await ensure_collection()
        logger.info("Qdrant collection ready")
    except Exception as e:
        logger.warning("Qdrant init failed (non-fatal): %s", e)

    yield

    # Clean up resources
    try:
        from app.services.llm.factory import close_all_clients
        await close_all_clients()
    except Exception as e:
        logger.warning("Error closing LLM clients: %s", e)

    try:
        from app.services.vector_store import close_qdrant_client
        await close_qdrant_client()
    except Exception as e:
        logger.warning("Error closing Qdrant client: %s", e)

    await close_db()
    await close_redis_connection()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI Search Engine for Indian users — real-time search, deep research, file intelligence, workspace collaboration, and Z+ security.",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware stack (order matters — outermost first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
app.add_exception_handler(Exception, global_exception_handler)


# ---------------------------------------------------------------------------
# Health Check — pings PostgreSQL + Redis
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health_check():
    checks = {"api": "ok"}

    # PostgreSQL (Supabase)
    try:
        from app.db.database import engine
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgresql"] = "ok"
    except Exception:
        checks["postgresql"] = "error"

    # Redis
    try:
        await redis_client.redis_pool.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "error"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    status_code = 200 if overall == "ok" else 503

    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={"status": overall, "version": settings.VERSION, "checks": checks},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
from app.api.routes import search, deep, workspace, actions, rag, agent, auth  # noqa: E402

app.include_router(search.router, prefix=settings.API_V1_STR + "/search", tags=["search"])
app.include_router(deep.router, prefix=settings.API_V1_STR + "/search/deep", tags=["deep_search"])
app.include_router(workspace.router, prefix=settings.API_V1_STR + "/workspace", tags=["workspace"])
app.include_router(actions.router, prefix=settings.API_V1_STR + "/actions", tags=["actions"])
app.include_router(rag.router, prefix=settings.API_V1_STR + "/rag", tags=["rag"])
app.include_router(agent.router, prefix=settings.API_V1_STR + "/agent", tags=["agent"])
app.include_router(auth.router, prefix=settings.API_V1_STR + "/auth", tags=["auth"])
