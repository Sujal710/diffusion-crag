"""Accounts and login: users live in SQLite, passwords are bcrypt-hashed, sessions are signed JWTs.

The middleware at the bottom protects every route except the public ones listed in PUBLIC_PATHS.
"""
import logging
import re
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import bcrypt
import jwt
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import get_settings

log = logging.getLogger("crag.auth")

DB_PATH = Path(__file__).parent / "data" / "users.db"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PUBLIC_PATHS = {"/", "/login", "/health", "/docs", "/openapi.json", "/redoc",
                "/auth/signup", "/auth/login"}
MAX_FAILS, FAIL_WINDOW_S = 5, 300

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL, "
        "password_hash TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    return con


def _get_user(email: str):
    with closing(_db()) as con:
        return con.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


# ---------------------------------------------------------------------------
# passwords and tokens
# ---------------------------------------------------------------------------
def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


@lru_cache
def _dummy_hash() -> str:
    return _hash("not-a-real-password")


@lru_cache
def _secret() -> str:
    s = get_settings().jwt_secret
    if s:
        return s
    log.warning("JWT_SECRET is not set: using a temporary secret, so every login ends when the server restarts. "
                "Add a long random JWT_SECRET to .env.")
    return secrets.token_urlsafe(48)


def create_token(user) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]), "email": user["email"], "name": user["name"],
        "iat": now, "exp": now + timedelta(minutes=get_settings().jwt_expire_minutes),
    }
    return jwt.encode(payload, _secret(), algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, _secret(), algorithms=["HS256"])  # raises on bad signature or expiry


# ---------------------------------------------------------------------------
# brute-force protection (in memory: resets when the server restarts)
# ---------------------------------------------------------------------------
_FAILS: dict = {}


def _check_rate(key: str) -> None:
    now = time.time()
    recent = [t for t in _FAILS.get(key, []) if now - t < FAIL_WINDOW_S]
    _FAILS[key] = recent
    if len(recent) >= MAX_FAILS:
        raise HTTPException(429, "Too many failed attempts. Try again in a few minutes.")


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=72)  # bcrypt only reads the first 72 bytes


class LoginRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=72)


def _session(user) -> dict:
    return {"token": create_token(user), "token_type": "bearer",
            "user": {"id": user["id"], "name": user["name"], "email": user["email"]}}


@router.post("/signup")
async def signup(req: SignupRequest):
    if not get_settings().allow_signup:
        raise HTTPException(403, "Sign-ups are closed.")
    email = req.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(422, "Enter a valid email address.")
    if len(req.password.encode()) > 72:
        raise HTTPException(422, "Password is too long (max 72 bytes).")
    try:
        with closing(_db()) as con, con:
            con.execute(
                "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (email, req.name.strip(), _hash(req.password), datetime.now(timezone.utc).isoformat()),
            )
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists. Log in instead.")
    return _session(_get_user(email))


@router.post("/login")
async def login(req: LoginRequest, request: Request):
    email = req.email.strip().lower()
    key = f"{request.client.host if request.client else '?'}|{email}"
    _check_rate(key)
    user = _get_user(email)
    # Always run one bcrypt check so a wrong email takes as long as a wrong password.
    ok = bcrypt.checkpw(req.password.encode(), (user["password_hash"] if user else _dummy_hash()).encode())
    if not (user and ok):
        _FAILS.setdefault(key, []).append(time.time())
        raise HTTPException(401, "Wrong email or password.")
    _FAILS.pop(key, None)
    return _session(user)


@router.get("/me")
async def me(request: Request):
    u = request.state.user
    return {"id": int(u["sub"]), "name": u["name"], "email": u["email"]}


# ---------------------------------------------------------------------------
# middleware: every route outside PUBLIC_PATHS needs "Authorization: Bearer <jwt>"
# ---------------------------------------------------------------------------
async def auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    def deny(message: str) -> JSONResponse:
        return JSONResponse({"detail": message}, status_code=401, headers={"WWW-Authenticate": "Bearer"})

    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        return deny("Please log in.")
    try:
        request.state.user = decode_token(token)
    except jwt.ExpiredSignatureError:
        return deny("Your session has expired. Please log in again.")
    except jwt.InvalidTokenError:
        return deny("Invalid login token. Please log in again.")
    return await call_next(request)
