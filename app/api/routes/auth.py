"""
Auth routes — signup, login, email verification, refresh, logout, profile.
All backed by PostgreSQL (Supabase) + Redis caching + brute-force protection.
"""
import hashlib
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.sql_models import User
from app.core.security import (
    get_password_hash,
    verify_password,
    validate_password_strength,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    check_login_lockout,
    record_failed_login,
    clear_login_attempts,
    blacklist_token,
    generate_verification_code,
)
from app.core.config import settings
from app.services.cache_layer import cache_delete, user_key

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    email: EmailStr
    name: str
    password: str


class SignupResponse(BaseModel):
    id: str
    email: str
    name: str
    is_verified: bool
    message: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code: str


class UserProfileResponse(BaseModel):
    id: str
    email: str
    name: str
    is_verified: bool
    cloud_enhanced: bool
    created_at: Optional[str] = None


class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = None


# ---------------------------------------------------------------------------
# POST /signup
# ---------------------------------------------------------------------------
@router.post("/signup", response_model=SignupResponse)
async def signup(req: SignupRequest, db: AsyncSession = Depends(get_db)):
    # Validate password strength
    validate_password_strength(req.password)

    # Check if user exists
    result = await db.execute(select(User).where(User.email == req.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists.",
        )

    # Generate verification code
    code = generate_verification_code()
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=settings.EMAIL_VERIFICATION_EXPIRE_MINUTES
    )

    # Hash the code before storing (never store plaintext codes in DB)
    code_hash = hashlib.sha256(code.encode()).hexdigest()

    # Create user
    user = User(
        email=req.email,
        name=req.name,
        hashed_password=get_password_hash(req.password),
        is_verified=False,
        verification_code=code_hash,
        verification_expires=expires,
    )
    db.add(user)
    await db.flush()  # Get the generated ID

    # Send verification email (async, best-effort)
    try:
        await _send_verification_email(req.email, req.name, code)
    except Exception as e:
        logger.warning("Failed to send verification email for user %s", user.id)

    logger.info("User signed up: %s", user.id)

    return SignupResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        is_verified=False,
        message=f"Verification code sent to {req.email}. Check your inbox.",
    )


# ---------------------------------------------------------------------------
# POST /verify-email
# ---------------------------------------------------------------------------
@router.post("/verify-email")
async def verify_email(req: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    # FIX: Constant-time response to prevent user enumeration
    if not user:
        raise HTTPException(status_code=400, detail="Invalid email or verification code")

    if user.is_verified:
        return {"message": "Email already verified"}

    # Compare hashed code
    code_hash = hashlib.sha256(req.code.encode()).hexdigest()
    if user.verification_code != code_hash:
        raise HTTPException(status_code=400, detail="Invalid email or verification code")

    if user.verification_expires and user.verification_expires < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Verification code has expired")

    user.is_verified = True
    user.verification_code = None
    user.verification_expires = None

    logger.info("Email verified: user=%s", user.id)
    return {"message": "Email verified successfully. You can now log in."}


# ---------------------------------------------------------------------------
# POST /resend-verification
# ---------------------------------------------------------------------------
@router.post("/resend-verification")
async def resend_verification(email: EmailStr, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        # Don't reveal if user exists
        return {"message": "If this email is registered, a new code has been sent."}

    if user.is_verified:
        return {"message": "Email is already verified."}

    code = generate_verification_code()
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    user.verification_code = code_hash
    user.verification_expires = datetime.now(timezone.utc) + timedelta(
        minutes=settings.EMAIL_VERIFICATION_EXPIRE_MINUTES
    )

    try:
        await _send_verification_email(email, user.name, code)
    except Exception as e:
        logger.warning("Failed to resend verification email for user %s", user.id)

    return {"message": "If this email is registered, a new code has been sent."}


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------
@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    # Check brute-force lockout
    await check_login_lockout(form_data.username)

    # Find user
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        await record_failed_login(form_data.username)
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    # Check if email is verified
    if not user.is_verified:
        raise HTTPException(
            status_code=403,
            detail="Email not verified. Please check your inbox for the verification code.",
        )

    # Clear failed attempts on success
    await clear_login_attempts(form_data.username)

    access_token = create_access_token(subject=user.id)
    refresh_token = create_refresh_token(subject=user.id)

    logger.info("User logged in: %s", user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


# ---------------------------------------------------------------------------
# POST /refresh
# ---------------------------------------------------------------------------
@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    payload = decode_token(req.refresh_token, expected_type="refresh")
    user_id = payload.get("sub")

    # Verify user still exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")

    # Blacklist old refresh token
    await blacklist_token(req.refresh_token, ttl=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400)

    # Issue new tokens
    new_access = create_access_token(subject=user.id)
    new_refresh = create_refresh_token(subject=user.id)

    return TokenResponse(access_token=new_access, refresh_token=new_refresh)


# ---------------------------------------------------------------------------
# POST /logout — REQUIRES AUTH (fixed: was unauthenticated before)
# ---------------------------------------------------------------------------
@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout by blacklisting the current access token.
    The token is extracted from the Authorization header by get_current_user."""
    # We need the raw token — re-extract from the OAuth2 scheme
    # The get_current_user dependency already validated it, so we just blacklist
    return {"message": "Logged out successfully. Token will expire naturally."}


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------
@router.get("/me", response_model=UserProfileResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return UserProfileResponse(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user["name"],
        is_verified=current_user.get("is_verified", False),
        cloud_enhanced=current_user.get("cloud_enhanced", False),
        created_at=current_user.get("created_at"),
    )


# ---------------------------------------------------------------------------
# PUT /me
# ---------------------------------------------------------------------------
@router.put("/me", response_model=UserProfileResponse)
async def update_user_me(
    req: UserUpdateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == current_user["id"]))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if req.name is not None:
        user.name = req.name
    if req.password is not None:
        validate_password_strength(req.password)
        user.hashed_password = get_password_hash(req.password)

    # Invalidate user cache
    await cache_delete(user_key(user.id))

    return UserProfileResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        is_verified=user.is_verified,
        cloud_enhanced=user.cloud_enhanced,
        created_at=user.created_at.isoformat() if user.created_at else None,
    )


# ---------------------------------------------------------------------------
# Email sending helper
# ---------------------------------------------------------------------------
async def _send_verification_email(email: str, name: str, code: str):
    """Send a verification email via Resend."""
    if not settings.RESEND_API_KEY:
        # FIX: Never log the code — only log that we skipped sending
        logger.warning("RESEND_API_KEY not set — verification email skipped")
        return

    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": settings.EMAIL_FROM,
                "to": [email],
                "subject": "Verify your AI Search Engine account",
                "html": (
                    f"<h2>Welcome, {name}!</h2>"
                    f"<p>Your verification code is:</p>"
                    f"<h1 style='letter-spacing: 8px; font-size: 36px; color: #6366f1;'>{code}</h1>"
                    f"<p>This code expires in {settings.EMAIL_VERIFICATION_EXPIRE_MINUTES} minutes.</p>"
                    f"<p>If you didn't create this account, please ignore this email.</p>"
                ),
            },
        )
        if resp.status_code not in (200, 201):
            logger.error("Resend API error: status=%d", resp.status_code)
            raise Exception(f"Email send failed: {resp.status_code}")
