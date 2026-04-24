"""
Security module — JWT tokens, password hashing, authentication, account lockout.
Z+ grade security: no hardcoded secrets, refresh tokens, brute-force protection.
"""
import jwt
import secrets
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Union, Dict, Optional

from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db
from app.db.redis_client import redis_client

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")


# ---------------------------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------------------------
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# Password Strength Validator
# ---------------------------------------------------------------------------
def validate_password_strength(password: str) -> None:
    """Enforce minimum password requirements."""
    errors = []
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        errors.append(f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter")
    if not re.search(r"\d", password):
        errors.append("Password must contain at least one digit")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append("Password must contain at least one special character")
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"password_errors": errors},
        )


# ---------------------------------------------------------------------------
# JWT Token Creation
# ---------------------------------------------------------------------------
def _get_secret() -> str:
    """Return the secret key. Raises if not configured."""
    if not settings.SECRET_KEY:
        raise RuntimeError(
            "SECRET_KEY is not set. Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return settings.SECRET_KEY


def create_access_token(subject: Union[str, Any], expires_delta: timedelta = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode = {"exp": expire, "sub": str(subject), "type": "access"}
    return jwt.encode(to_encode, _get_secret(), algorithm=settings.ALGORITHM)


def create_refresh_token(subject: Union[str, Any]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    return jwt.encode(to_encode, _get_secret(), algorithm=settings.ALGORITHM)


def decode_token(token: str, expected_type: str = "access") -> Dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[settings.ALGORITHM])
        if payload.get("type") != expected_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {expected_type}.",
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


# ---------------------------------------------------------------------------
# Brute-Force Protection (Redis-backed)
# ---------------------------------------------------------------------------
async def check_login_lockout(email: str) -> None:
    """Check if account is locked out due to too many failed attempts."""
    try:
        pool = redis_client.redis_pool
        if not pool:
            return
        key = f"login_attempts:{email}"
        attempts = await pool.get(key)
        if attempts and int(attempts) >= settings.LOGIN_MAX_ATTEMPTS:
            ttl = await pool.ttl(key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Account locked due to too many failed attempts. Try again in {max(ttl, 0)} seconds.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # Fail-open if Redis is down


async def record_failed_login(email: str) -> None:
    """Increment failed login counter."""
    try:
        pool = redis_client.redis_pool
        if not pool:
            return
        key = f"login_attempts:{email}"
        current = await pool.incr(key)
        if current == 1:
            await pool.expire(key, settings.LOGIN_LOCKOUT_MINUTES * 60)
    except Exception:
        pass


async def clear_login_attempts(email: str) -> None:
    """Clear failed login counter on successful login."""
    try:
        pool = redis_client.redis_pool
        if pool:
            await pool.delete(f"login_attempts:{email}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Token Blacklist (for logout)
# ---------------------------------------------------------------------------
async def blacklist_token(token: str, ttl: int = None) -> None:
    """Add a token to the blacklist in Redis."""
    try:
        pool = redis_client.redis_pool
        if not pool:
            return
        if ttl is None:
            ttl = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        await pool.setex(f"blacklist:{token}", ttl, "1")
    except Exception:
        pass


async def is_token_blacklisted(token: str) -> bool:
    """Check if a token has been revoked."""
    try:
        pool = redis_client.redis_pool
        if not pool:
            return False
        result = await pool.get(f"blacklist:{token}")
        return result is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Email Verification Code Generation
# ---------------------------------------------------------------------------
def generate_verification_code() -> str:
    """Generate a 6-digit verification code."""
    return f"{secrets.randbelow(900000) + 100000}"


# ---------------------------------------------------------------------------
# Current User Dependency (PostgreSQL)
# ---------------------------------------------------------------------------
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Dict:
    """FastAPI dependency: extract and validate the current user from JWT."""
    # Check if token is blacklisted
    if await is_token_blacklisted(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )

    payload = decode_token(token, expected_type="access")
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    # Check Redis cache first
    cached_user = await _get_cached_user(user_id)
    if cached_user:
        return cached_user

    # Fallback to DB
    from app.db.sql_models import User
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    user_dict = {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "is_verified": user.is_verified,
        "cloud_enhanced": user.cloud_enhanced,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }

    # Cache for future requests
    await _cache_user(user_id, user_dict)
    return user_dict


async def _get_cached_user(user_id: str) -> Optional[Dict]:
    """Get user from Redis cache."""
    try:
        import json
        pool = redis_client.redis_pool
        if not pool:
            return None
        cached = await pool.get(f"user:{user_id}")
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


async def _cache_user(user_id: str, user_dict: Dict) -> None:
    """Cache user profile in Redis."""
    try:
        import json
        pool = redis_client.redis_pool
        if pool:
            await pool.setex(
                f"user:{user_id}",
                settings.REDIS_USER_TTL,
                json.dumps(user_dict, default=str),
            )
    except Exception:
        pass
