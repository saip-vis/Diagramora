"""Supabase client helpers for authentication and user-scoped database access."""

from __future__ import annotations

import os

from supabase import Client, create_client


class SupabaseConfigurationError(RuntimeError):
    """Raised when required Supabase environment variables are missing."""


def _settings() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
    if not url or not key:
        raise SupabaseConfigurationError(
            "Supabase is not configured. Fill SUPABASE_URL and "
            "SUPABASE_PUBLISHABLE_KEY in .env."
        )
    return url, key


def public_client() -> Client:
    """Return a fresh client for public Auth operations.

    Supabase auth calls mutate the client's in-memory session, so this client
    must never be cached or shared between concurrent users.
    """
    url, key = _settings()
    return create_client(url, key)


def user_client(access_token: str, refresh_token: str) -> Client:
    """Return a user-scoped client whose database calls respect RLS policies."""
    url, key = _settings()
    client = create_client(url, key)
    client.auth.set_session(access_token, refresh_token)
    return client
