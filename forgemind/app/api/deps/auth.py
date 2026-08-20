"""Bearer-token auth for the mutating Task API routes (Phase 10.5).

A SINGLE shared-secret token gates every state-mutating route (POST /tasks,
cancel, approve, reject) — single-operator scope, no user accounts, no OAuth,
no sessions. Read-only routes (``GET /tasks``, ``GET /tasks/{id}``,
``GET /events``) and ``/health`` stay open so a human can watch a task walk
the pipeline without needing the token.

The comparison is constant-time (``secrets.compare_digest``) and every
non-credential branch short-circuits BEFORE any ``.encode()`` so a missing or
empty credential can never reach the digest compare or crash on ``None``.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings

# The audit identity for token-authenticated actions. There are no user
# accounts — "who" is simply "the token holder".
TOKEN_HOLDER = "token-holder"

_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_api_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Return ``TOKEN_HOLDER`` if the bearer token is valid, else raise 401.

    Ordering is deliberate: ``credentials is None`` (no ``Authorization``
    header at all) short-circuits before any comparison, and the empty-credential
    guard runs before ``.encode()``/``compare_digest`` so a falsy value never
    reaches the digest path.
    """
    if credentials is None:
        raise _unauthorized("Missing bearer token")
    if credentials.scheme.lower() != "bearer":
        raise _unauthorized("Invalid authorization scheme")
    expected = get_settings().api_token
    if not expected:
        # Defense in depth: the config validator fails closed in production
        # (None or "" both refuse startup), so this is unreachable there — but
        # never silently run unauthenticated if it is somehow reached.
        raise _unauthorized("Server not configured for token auth")
    if not credentials.credentials:
        raise _unauthorized("Empty bearer token")
    if not secrets.compare_digest(
        credentials.credentials.encode("utf-8"), expected.encode("utf-8")
    ):
        raise _unauthorized("Invalid bearer token")
    return TOKEN_HOLDER
