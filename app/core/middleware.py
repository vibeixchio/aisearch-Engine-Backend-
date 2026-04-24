import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.db.redis_client import redis_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Request ID + Timing Middleware
# ---------------------------------------------------------------------------
class RequestContextMiddleware(BaseHTTPMiddleware):
    """Injects X-Request-ID header and logs request timing."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            logger.exception(
                "Unhandled exception",
                extra={"request_id": request_id, "path": str(request.url.path)},
            )
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id},
            )

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
        return response


# ---------------------------------------------------------------------------
# 2. Rate Limiting Middleware (Redis-backed)
# ---------------------------------------------------------------------------
RATE_LIMIT_FREE_REQUESTS = 60    # per window
RATE_LIMIT_PRO_REQUESTS = 300    # per window
RATE_LIMIT_WINDOW_SECONDS = 60

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Adaptive sliding-window rate limiter using Redis."""

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and docs
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Default limits
        limit = RATE_LIMIT_FREE_REQUESTS
        client_id = request.client.host if request.client else "unknown"

        # Try to identify user tier from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            from app.core.security import decode_token, _get_cached_user
            try:
                payload = decode_token(token, expected_type="access")
                user_id = payload.get("sub")
                if user_id:
                    client_id = f"user:{user_id}"
                    # Check if Pro tier
                    cached_user = await _get_cached_user(user_id)
                    if cached_user and cached_user.get("cloud_enhanced"):
                        limit = RATE_LIMIT_PRO_REQUESTS
            except Exception:
                pass  # Fallback to IP-based free tier limit if token is invalid

        key = f"ratelimit:{client_id}"

        try:
            pool = redis_client.redis_pool
            if pool:
                current = await pool.incr(key)
                if current == 1:
                    await pool.expire(key, RATE_LIMIT_WINDOW_SECONDS)
                if current > limit:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "limit": limit,
                            "retry_after_seconds": RATE_LIMIT_WINDOW_SECONDS,
                        },
                    )
        except Exception:
            # If Redis is down, allow the request through (fail-open)
            pass

        return await call_next(request)


# ---------------------------------------------------------------------------
# 3. Global Exception Handler
# ---------------------------------------------------------------------------
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "Unhandled error",
        extra={"request_id": request_id, "path": str(request.url.path)},
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": request_id},
    )
