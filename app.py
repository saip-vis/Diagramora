"""Diagramora Flask application entry point."""

import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file, session
from flask_session import Session
from dotenv import load_dotenv

load_dotenv()

from ai_service import (
    collect_ai_usage,
    extract_flowchart_json,
    finalize_graph,
    generate_title,
    moderate_text,
    transcribe_audio,
    update_graph_live,
    edit_flowchart_with_prompt,
)
from graph_model import Flowchart
from layout_engine import build_layout_plan
from renderer import SUPPORTED_FORMATS, render_flowchart, sanitize_filename
from supabase_client import SupabaseConfigurationError, public_client, user_client

app = Flask(__name__, template_folder=".")
_production = os.getenv("APP_ENV") == "production"
_secret_key = os.getenv("FLASK_SECRET_KEY")
if _production and not _secret_key:
    raise RuntimeError("FLASK_SECRET_KEY must be configured in production.")
app.secret_key = _secret_key or "change-this-development-secret"
app.config.update(
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR=os.getenv("SESSION_FILE_DIR", str(Path(__file__).with_name(".flask_session"))),
    SESSION_PERMANENT=False,
    SESSION_USE_SIGNER=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=_production,
    SESSION_REFRESH_EACH_REQUEST=False,
)
Session(app)


@app.get("/healthz")
def healthz():
    """Lightweight host health check that does not call external services."""
    return jsonify({"status": "ok"})

MAX_DESIGNS_PER_USER = 3
MAX_TRANSCRIPT_CHARS = 20_000
MAX_INSTRUCTION_CHARS = 1_000
MAX_GRAPH_BYTES = 300_000
MAX_GRAPH_NODES = 100
MAX_GRAPH_EDGES = 250
MAX_GESTURES = 250
DAILY_AI_LIMIT = max(10, min(500, int(os.getenv("DAILY_AI_LIMIT", "200"))))
AI_LIMITS = {
    "generate": (3600, 5),
    "live_update": (60, 35),
    "live_title": (60, 12),
    "finalize": (600, 8),
    "ai_edit": (600, 12),
    "export": (600, 15),
}
# Standard-processing estimates in USD per one million tokens. Raw token counts
# remain authoritative if prices change; this table is only an operational estimate.
MODEL_PRICING_USD_PER_MILLION = {
    "gpt-4.1-nano": {"input": 0.10, "cached": 0.025, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "cached": 0.075, "output": 0.60},
    "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
}
_public_attempts = defaultdict(deque)
_public_attempts_lock = threading.Lock()


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(self), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; "
        "connect-src 'self' https://cdn.jsdelivr.net https://storage.googleapis.com; "
        "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    )
    if request.path != "/":
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.before_request
def reject_cross_site_mutations():
    if request.method not in {"POST", "PATCH", "PUT", "DELETE"}:
        return None
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        return jsonify({"error": "Cross-site request rejected."}), 403
    origin = request.headers.get("Origin")
    if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
        return jsonify({"error": "Request origin is not allowed."}), 403
    return None


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request is too large. Shorten the recording or flowchart and try again."}), 413


def _public_rate_limited(scope: str, limit: int, window_seconds: int, *, record_attempt: bool = True) -> bool:
    """Check a local unauthenticated rate limit and optionally record the attempt."""
    key = (scope, request.remote_addr or "unknown")
    now = time.monotonic()
    with _public_attempts_lock:
        attempts = _public_attempts[key]
        while attempts and now - attempts[0] >= window_seconds:
            attempts.popleft()
        if len(attempts) >= limit:
            return True
        if record_attempt:
            attempts.append(now)
    return False


def _server_error(public_message: str):
    logging.exception(public_message)
    return jsonify({"error": public_message}), 500


def _first_row(data):
    if isinstance(data, list):
        return data[0] if data else None
    return data if isinstance(data, dict) else None


def _bounded_text(value, field: str, maximum: int, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{field} is required.")
    if len(text) > maximum:
        raise ValueError(f"{field} is too long (maximum {maximum:,} characters).")
    return text


def _validated_gestures(value) -> list:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_GESTURES:
        raise ValueError(f"Gestures must be a list with at most {MAX_GESTURES} items.")
    cleaned = []
    for item in value:
        if not isinstance(item, dict):
            continue
        gesture = _bounded_text(item.get("gesture"), "Gesture name", 80, required=True)
        try:
            timestamp = max(0.0, min(float(item.get("time", 0)), 86_400.0))
        except (TypeError, ValueError):
            timestamp = 0.0
        cleaned.append({"gesture": gesture, "time": timestamp})
    return cleaned


def _validated_graph(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Flowchart data must be an object.")
    if len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) > MAX_GRAPH_BYTES:
        raise ValueError("Flowchart data is too large.")
    nodes = value.get("nodes", [])
    edges = value.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Flowchart nodes and edges must be lists.")
    if len(nodes) > MAX_GRAPH_NODES or len(edges) > MAX_GRAPH_EDGES:
        raise ValueError(f"Flowcharts support up to {MAX_GRAPH_NODES} nodes and {MAX_GRAPH_EDGES} connections.")
    parsed = Flowchart.from_dict(value)
    ids = [node.id for node in parsed.nodes]
    if any(not node_id for node_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Every flowchart node must have a unique ID.")
    if any(not node.label or len(node.label) > 200 for node in parsed.nodes):
        raise ValueError("Every node needs a label of at most 200 characters.")
    valid_ids = set(ids)
    if any(edge.source not in valid_ids or edge.target not in valid_ids for edge in parsed.edges):
        raise ValueError("Every connection must reference existing nodes.")
    if any(len(edge.label) > 120 for edge in parsed.edges):
        raise ValueError("Connection labels must be at most 120 characters.")
    return parsed.to_dict()


def _consume_ai_quota(operation: str):
    client = _authenticated_client()
    if not client:
        return jsonify({"error": "Login required."}), 401
    window_seconds, limit = AI_LIMITS[operation]
    try:
        total = client.rpc("consume_ai_quota", {
            "p_operation": "ai_total", "p_window_seconds": 86_400, "p_limit": DAILY_AI_LIMIT,
        }).execute()
        if not total.data:
            return jsonify({"error": "Daily AI usage limit reached. Try again tomorrow.", "code": "daily_ai_limit"}), 429
        scoped = client.rpc("consume_ai_quota", {
            "p_operation": operation, "p_window_seconds": window_seconds, "p_limit": limit,
        }).execute()
        if not scoped.data:
            return jsonify({"error": "You're doing that too quickly. Wait a moment and try again.", "code": "rate_limited"}), 429
    except Exception:
        logging.exception("AI quota check failed")
        return jsonify({"error": "Usage protection is not configured. Run the latest Supabase schema."}), 503
    return None


def _estimated_cost_microusd(usage: dict) -> int:
    model = str(usage.get("model") or "")
    pricing = next((rates for prefix, rates in MODEL_PRICING_USD_PER_MILLION.items() if model.startswith(prefix)), None)
    if not pricing:
        return 0
    input_tokens = max(int(usage.get("input_tokens") or 0), 0)
    cached_tokens = min(max(int(usage.get("cached_input_tokens") or 0), 0), input_tokens)
    uncached_tokens = input_tokens - cached_tokens
    output_tokens = max(int(usage.get("output_tokens") or 0), 0)
    # USD / 1M tokens is numerically equal to micro-USD / token.
    return round(
        uncached_tokens * pricing["input"]
        + cached_tokens * pricing["cached"]
        + output_tokens * pricing["output"]
    )


def track_ai_usage(operation: str):
    """Persist provider token telemetry without making the user request depend on it."""
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            with collect_ai_usage() as usage_records:
                try:
                    return view(*args, **kwargs)
                finally:
                    if usage_records:
                        client = _authenticated_client()
                        if client:
                            for usage in usage_records:
                                try:
                                    client.rpc("record_ai_provider_usage", {
                                        "p_operation": operation,
                                        "p_model": usage.get("model") or "unknown",
                                        "p_input_tokens": usage.get("input_tokens", 0),
                                        "p_cached_input_tokens": usage.get("cached_input_tokens", 0),
                                        "p_output_tokens": usage.get("output_tokens", 0),
                                        "p_estimated_cost_microusd": _estimated_cost_microusd(usage),
                                        "p_provider_request_id": usage.get("request_id"),
                                    }).execute()
                                except Exception:
                                    logging.exception("Could not record AI provider usage")
        return wrapped
    return decorator


def _reserve_entitlement(client, operation: str):
    try:
        result = client.rpc("reserve_ai_entitlement", {"p_operation": operation}).execute()
        entitlement = result.data or {}
        if entitlement.get("allowed"):
            return entitlement, None
        noun = "flowchart generations" if operation == "generation" else "AI edits"
        return None, (jsonify({
            "error": f"You've used all free {noun}. Your existing designs and manual editing are still available.",
            "code": entitlement.get("code") or "trial_exhausted",
        }), 402)
    except Exception:
        logging.exception("AI entitlement check failed")
        return None, (jsonify({"error": "Trial usage is not configured. Run the latest Supabase schema."}), 503)


def _settle_entitlement(client, reservation_id: str | None):
    if not reservation_id:
        return
    try:
        client.rpc("settle_ai_entitlement", {"p_reservation_id": reservation_id}).execute()
    except Exception:
        logging.exception("Could not settle AI entitlement reservation")


def _refund_entitlement(client, reservation_id: str | None):
    if not reservation_id:
        return
    try:
        client.rpc("refund_ai_entitlement", {"p_reservation_id": reservation_id}).execute()
    except Exception:
        logging.exception("Could not refund failed AI entitlement")


def _require_entitlement_available(client, operation: str):
    """Block further provider spend once a free allowance has been exhausted."""
    try:
        result = client.rpc("get_ai_entitlements").execute()
        entitlement = result.data or {}
        if entitlement.get("plan", "free") != "free":
            return None
        remaining_key = "generations_remaining" if operation == "generation" else "edits_remaining"
        if int(entitlement.get(remaining_key, 0)) > 0:
            return None
        code = "generation_trial_exhausted" if operation == "generation" else "edit_trial_exhausted"
        return jsonify({
            "error": "Your free AI allowance has been used. Existing designs and manual editing are still available.",
            "code": code,
        }), 402
    except Exception:
        logging.exception("Could not read AI entitlements")
        return jsonify({"error": "Trial usage is not configured. Run the latest Supabase schema."}), 503


def charge_entitlement(operation: str):
    """Charge a legacy one-shot AI route, refunding any unsuccessful response."""
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            client = _authenticated_client()
            entitlement, entitlement_error = _reserve_entitlement(client, operation)
            if entitlement_error:
                return entitlement_error
            reservation_id = entitlement.get("reservation_id")
            try:
                response = view(*args, **kwargs)
                status_code = response[1] if isinstance(response, tuple) else getattr(response, "status_code", 200)
                if int(status_code) >= 400:
                    _refund_entitlement(client, reservation_id)
                else:
                    _settle_entitlement(client, reservation_id)
                return response
            except Exception:
                _refund_entitlement(client, reservation_id)
                raise
        return wrapped
    return decorator


def _auth_tokens():
    access_token = session.get("supabase_access_token")
    refresh_token = session.get("supabase_refresh_token")
    if not access_token or not refresh_token:
        return None
    return access_token, refresh_token


def _clear_and_rotate_session():
    """Clear auth state and prevent older concurrent requests reusing its session ID."""
    regenerate = getattr(app.session_interface, "regenerate", None)
    if callable(regenerate):
        # Flask-Session only regenerates a non-empty session. Authentication
        # sessions are normally non-empty, while the marker covers reset flows.
        if not session:
            session["_rotation_marker"] = True
        regenerate(session)
    session.clear()


def _authenticated_client():
    tokens = _auth_tokens()
    if not tokens:
        return None
    try:
        client = user_client(*tokens)
        verified = client.auth.get_user(tokens[0])
        if not getattr(verified, "user", None):
            _clear_and_rotate_session()
            return None
        refreshed = client.auth.get_session()
        if refreshed:
            if session.get("supabase_access_token") != refreshed.access_token:
                session["supabase_access_token"] = refreshed.access_token
            if session.get("supabase_refresh_token") != refreshed.refresh_token:
                session["supabase_refresh_token"] = refreshed.refresh_token
        return client
    except Exception:
        _clear_and_rotate_session()
        return None


def _user_payload(client):
    tokens = _auth_tokens()
    if not client or not tokens:
        return None
    response = client.auth.get_user(tokens[0])
    user = getattr(response, "user", None)
    if not user:
        return None
    metadata = user.user_metadata or {}
    return {
        "id": str(user.id),
        "email": user.email,
        "username": metadata.get("username") or (user.email or "user").split("@")[0],
    }


def login_required(view):
    """Require a valid Supabase session for protected application features."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not _authenticated_client():
            return jsonify({"error": "Login required."}), 401
        return view(*args, **kwargs)

    return wrapped


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/favicon.svg")
def favicon():
    return send_file(Path(__file__).with_name("favicon.svg"), mimetype="image/svg+xml", max_age=86400)


@app.get("/robots.txt")
def robots():
    base_url = request.url_root.rstrip("/")
    body = f"User-agent: *\nAllow: /\nSitemap: {base_url}/sitemap.xml\n"
    return Response(body, mimetype="text/plain")


@app.get("/sitemap.xml")
def sitemap():
    base_url = request.url_root.rstrip("/")
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{base_url}/</loc></url>\n"
        "</urlset>\n"
    )
    return Response(body, mimetype="application/xml")


@app.errorhandler(404)
def not_found(_error):
    if request.accept_mimetypes.best == "application/json":
        return jsonify({"error": "Not found."}), 404
    return render_template("404.html"), 404


@app.route("/auth/status")
def auth_status():
    client = _authenticated_client()
    user = _user_payload(client) if client else None
    return jsonify({"logged_in": bool(user), "user": user})


@app.route("/signup", methods=["POST"])
def signup():
    if _public_rate_limited("signup", 5, 3600):
        return jsonify({"error": "Too many signup attempts. Try again later."}), 429
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    username = str(data.get("username", "")).strip()
    if not email or not password or len(email) > 254:
        return jsonify({"error": "Email and password are required."}), 400
    if len(password) < 10 or len(password) > 128:
        return jsonify({"error": "Password must be between 10 and 128 characters."}), 400
    if username and (len(username) > 60 or not re.fullmatch(r"[A-Za-z0-9_.-]+", username)):
        return jsonify({"error": "Username may contain letters, numbers, periods, underscores, and hyphens."}), 400
    try:
        response = public_client().auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"username": username or email.split("@")[0]}},
        })
        auth_session = getattr(response, "session", None)
        if auth_session:
            _clear_and_rotate_session()
            session["supabase_access_token"] = auth_session.access_token
            session["supabase_refresh_token"] = auth_session.refresh_token
        created_user = getattr(response, "user", None)
        created_metadata = (getattr(created_user, "user_metadata", None) or {}) if created_user else {}
        return jsonify({
            "message": "Account created. Check your email to confirm it before logging in."
                if not auth_session else "Account created.",
            "logged_in": bool(auth_session),
            "user": ({
                "id": str(created_user.id),
                "email": created_user.email,
                "username": created_metadata.get("username") or email.split("@")[0],
            } if auth_session and created_user else None),
        }), 201
    except SupabaseConfigurationError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logging.exception("Signup failed")
        return jsonify({"error": "Account creation failed. The email or username may already be in use."}), 400


@app.route("/login", methods=["POST"])
def login():
    # Successful account switches should not consume the failed-login budget.
    if _public_rate_limited("login", 12, 900, record_attempt=False):
        return jsonify({"error": "Too many login attempts. Wait a few minutes and try again."}), 429
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    if not email or not password or len(email) > 254 or len(password) > 128:
        return jsonify({"error": "Email and password are required."}), 400
    try:
        response = public_client().auth.sign_in_with_password({"email": email, "password": password})
        auth_session = response.session
        _clear_and_rotate_session()
        session["supabase_access_token"] = auth_session.access_token
        session["supabase_refresh_token"] = auth_session.refresh_token
        user = response.user
        metadata = user.user_metadata or {}
        return jsonify({
            "logged_in": True,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "username": metadata.get("username") or email.split("@")[0],
            },
        })
    except SupabaseConfigurationError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        _public_rate_limited("login", 12, 900)
        return jsonify({"error": "Invalid email or password, or email is not confirmed."}), 401


@app.route("/logout", methods=["POST"])
def logout():
    client = _authenticated_client()
    if client:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    _clear_and_rotate_session()
    return jsonify({"logged_in": False})




@app.route("/auth/reset-password", methods=["POST"])
def reset_password():
    if _public_rate_limited("password_reset", 4, 3600):
        return jsonify({"message": "If an account exists for that email, a reset link will be sent when allowed."})
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    if not email:
        return jsonify({"error": "Email is required."}), 400
    try:
        base_url = os.getenv("APP_BASE_URL", request.url_root).strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise SupabaseConfigurationError("APP_BASE_URL must start with http:// or https://")
        redirect_url = base_url + "/?reset=1"
        public_client().auth.reset_password_for_email(email, {"redirect_to": redirect_url})
        return jsonify({"message": "If an account exists for that email, a reset link has been sent."})
    except SupabaseConfigurationError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        return jsonify({"message": "If an account exists for that email, a reset link has been sent."})


@app.route("/auth/complete-reset", methods=["POST"])
def complete_password_reset():
    if _public_rate_limited("complete_password_reset", 8, 900):
        return jsonify({"error": "Too many attempts. Wait a few minutes and try again."}), 429
    data = request.get_json(silent=True) or {}
    access_token = str(data.get("access_token") or "")
    refresh_token = str(data.get("refresh_token") or "")
    password = str(data.get("password") or "")
    if not access_token or not refresh_token:
        return jsonify({"error": "This password-reset link is invalid or expired."}), 400
    if len(password) < 10 or len(password) > 128:
        return jsonify({"error": "Password must be between 10 and 128 characters."}), 400
    try:
        client = user_client(access_token, refresh_token)
        verified = client.auth.get_user(access_token)
        if not getattr(verified, "user", None):
            return jsonify({"error": "This password-reset link is invalid or expired."}), 400
        client.auth.update_user({"password": password})
        _clear_and_rotate_session()
        return jsonify({"message": "Password updated. You can now log in."})
    except Exception:
        logging.exception("Password reset completion failed")
        return jsonify({"error": "The reset link is invalid or expired. Request a new one."}), 400


@app.route("/profile", methods=["GET", "PATCH"])
def profile():
    client = _authenticated_client()
    user = _user_payload(client) if client else None
    if not client or not user:
        return jsonify({"error": "Login required."}), 401
    try:
        if request.method == "GET":
            result = client.table("profiles").select("id,username,display_name,plan,created_at,updated_at").eq("id", user["id"]).execute()
            profile_data = result.data[0] if result.data else {
                "id": user["id"], "username": user["username"], "display_name": user["username"],
                "plan": "free",
            }
            profile_data["email"] = user["email"]
            entitlement_result = client.rpc("get_ai_entitlements").execute()
            return jsonify({"profile": profile_data, "entitlements": entitlement_result.data or {}})

        data = request.get_json(silent=True) or {}
        username = str(data.get("username", "")).strip()
        display_name = str(data.get("display_name", "")).strip()
        if not username:
            return jsonify({"error": "Username is required."}), 400
        if len(username) > 60 or not re.fullmatch(r"[A-Za-z0-9_.-]+", username):
            return jsonify({"error": "Username may contain letters, numbers, periods, underscores, and hyphens."}), 400
        payload = {
            "username": username[:60],
            "display_name": (display_name or username)[:100],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        result = client.table("profiles").update(payload).eq("id", user["id"]).select("id,username,display_name,plan,updated_at").execute()
        if not result.data:
            return jsonify({"error": "Profile was not updated."}), 500
        return jsonify({"profile": result.data[0]})
    except Exception:
        return _server_error("Could not access profile.")


@app.route("/designs", methods=["GET", "POST"])
def designs():
    client = _authenticated_client()
    user = _user_payload(client) if client else None
    if not client or not user:
        return jsonify({"error": "Login required."}), 401

    if request.method == "GET":
        try:
            result = (
                client.table("designs")
                .select("id,title,workflow_json,created_at,updated_at")
                .eq("user_id", user["id"])
                .order("updated_at", desc=True)
                .execute()
            )
            designs_data = []
            for row in result.data or []:
                design = dict(row)
                workflow = design.pop("workflow_json", {}) or {}
                design["preview_graph"] = workflow.get("graph", {"nodes": [], "edges": []})
                designs_data.append(design)
            used = len(designs_data)
            return jsonify({
                "designs": designs_data,
                "usage": {
                    "used": used,
                    "limit": MAX_DESIGNS_PER_USER,
                    "remaining": max(0, MAX_DESIGNS_PER_USER - used),
                },
            })
        except Exception:
            return _server_error("Could not load designs.")

    data = request.get_json(silent=True) or {}
    design_id = data.get("id")
    try:
        graph = _validated_graph(data.get("graph") or {"nodes": [], "edges": []})
        transcript = _bounded_text(data.get("transcript"), "Transcript", MAX_TRANSCRIPT_CHARS)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    payload = {
        "user_id": user["id"],
        "title": str(data.get("title") or "Untitled Flowchart")[:200],
        "workflow_json": {
            "graph": graph,
            "transcript": transcript,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if design_id:
            result = (
                client.table("designs")
                .update(payload)
                .eq("id", str(design_id))
                .eq("user_id", user["id"])
                .select("id,title,created_at,updated_at")
                .execute()
            )
        else:
            result = (
                client.rpc("create_design_limited", {
                    "p_title": payload["title"],
                    "p_workflow_json": payload["workflow_json"],
                })
                .execute()
            )
        saved_row = _first_row(result.data)
        if not saved_row:
            return jsonify({"error": "Design was not saved."}), 500
        return jsonify({"design": saved_row})
    except Exception as exc:
        if "design_limit_reached" in str(exc):
            return jsonify({
                "error": f"Private testing accounts can save up to {MAX_DESIGNS_PER_USER} designs. Delete one to save another.",
                "code": "design_limit_reached", "limit": MAX_DESIGNS_PER_USER,
            }), 403
        return _server_error("Could not save design.")


@app.route("/feedback", methods=["POST"])
@login_required
def submit_feedback():
    data = request.get_json(silent=True) or {}
    try:
        rating = int(data.get("rating"))
    except (TypeError, ValueError):
        return jsonify({"error": "Choose a rating from 1 to 5."}), 400
    try:
        message = _bounded_text(data.get("message"), "Feedback", 2_000, required=True)
        context = _bounded_text(data.get("context") or "general", "Feedback context", 80)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if rating < 1 or rating > 5:
        return jsonify({"error": "Choose a rating from 1 to 5."}), 400
    client = _authenticated_client()
    try:
        result = client.rpc("submit_feedback", {
            "p_rating": rating,
            "p_message": message,
            "p_context": context or "general",
        }).execute()
        return jsonify({"submitted": True, "id": result.data}), 201
    except Exception as exc:
        if "feedback_limit_reached" in str(exc):
            return jsonify({"error": "You've sent several responses today. Please try again tomorrow."}), 429
        return _server_error("Could not send feedback.")


@app.route("/designs/<design_id>", methods=["GET", "PATCH", "DELETE"])
def design_detail(design_id):
    client = _authenticated_client()
    user = _user_payload(client) if client else None
    if not client or not user:
        return jsonify({"error": "Login required."}), 401
    try:
        if request.method == "DELETE":
            owned = client.table("designs").select("id").eq("id", design_id).eq("user_id", user["id"]).execute()
            if not owned.data:
                return jsonify({"error": "Design not found."}), 404
            client.table("designs").delete().eq("id", design_id).eq("user_id", user["id"]).execute()
            return jsonify({"deleted": True})
        if request.method == "PATCH":
            data = request.get_json(silent=True) or {}
            title = str(data.get("title") or "Untitled Flowchart").strip()[:200]
            result = (
                client.table("designs")
                .update({"title": title, "updated_at": datetime.now(timezone.utc).isoformat()})
                .eq("id", design_id)
                .eq("user_id", user["id"])
                .select("id,title,created_at,updated_at")
                .execute()
            )
            if not result.data:
                return jsonify({"error": "Design not found."}), 404
            return jsonify({"design": result.data[0]})
        result = (
            client.table("designs")
            .select("id,title,workflow_json,created_at,updated_at")
            .eq("id", design_id)
            .eq("user_id", user["id"])
            .execute()
        )
        if not result.data:
            return jsonify({"error": "Design not found."}), 404
        row = result.data[0]
        workflow = row.pop("workflow_json", {}) or {}
        row["graph"] = workflow.get("graph", {"nodes": [], "edges": []})
        row["transcript"] = workflow.get("transcript", "")
        return jsonify({"design": row})
    except Exception:
        return _server_error("Could not access design.")




@app.route("/designs/<design_id>/duplicate", methods=["POST"])
def duplicate_design(design_id):
    client = _authenticated_client()
    user = _user_payload(client) if client else None
    if not client or not user:
        return jsonify({"error": "Login required."}), 401
    try:
        created = client.rpc("duplicate_design_limited", {"p_design_id": design_id}).execute()
        created_row = _first_row(created.data)
        if not created_row:
            return jsonify({"error": "Design not found."}), 404
        return jsonify({"design": created_row}), 201
    except Exception as exc:
        if "design_limit_reached" in str(exc):
            return jsonify({
                "error": f"Private testing accounts can save up to {MAX_DESIGNS_PER_USER} designs. Delete one before duplicating.",
                "code": "design_limit_reached", "limit": MAX_DESIGNS_PER_USER,
            }), 403
        if "design_not_found" in str(exc):
            return jsonify({"error": "Design not found."}), 404
        return _server_error("Could not duplicate design.")


@app.route("/designs/<design_id>/versions", methods=["GET", "POST"])
def design_versions(design_id):
    client = _authenticated_client()
    user = _user_payload(client) if client else None
    if not client or not user:
        return jsonify({"error": "Login required."}), 401
    try:
        if request.method == "GET":
            result = client.table("design_versions").select("id,version_number,created_at").eq("design_id", design_id).eq("user_id", user["id"]).order("version_number", desc=True).limit(20).execute()
            return jsonify({"versions": result.data or []})
        created = client.rpc("create_design_version", {"p_design_id": design_id}).execute()
        created_row = _first_row(created.data)
        if not created_row:
            return jsonify({"error": "Version was not created."}), 500
        return jsonify({"version": created_row}), 201
    except Exception as exc:
        if "design_not_found" in str(exc):
            return jsonify({"error": "Design not found."}), 404
        if "version_limit_reached" in str(exc):
            return jsonify({
                "error": "Free accounts can keep up to 20 versions per design.",
                "code": "version_limit_reached",
                "limit": 20,
            }), 403
        return _server_error("Could not access versions.")


@app.route("/versions/<version_id>", methods=["GET"])
def get_version(version_id):
    client = _authenticated_client()
    user = _user_payload(client) if client else None
    if not client or not user:
        return jsonify({"error": "Login required."}), 401
    try:
        result = client.table("design_versions").select("id,design_id,version_number,workflow_json,created_at").eq("id", version_id).eq("user_id", user["id"]).execute()
        if not result.data:
            return jsonify({"error": "Version not found."}), 404
        row = result.data[0]
        workflow = row.pop("workflow_json", {}) or {}
        row["graph"] = workflow.get("graph", {"nodes": [], "edges": []})
        row["transcript"] = workflow.get("transcript", "")
        return jsonify({"version": row})
    except Exception:
        return _server_error("Could not load version.")


@app.route("/generate", methods=["POST"])
@login_required
@charge_entitlement("generation")
@track_ai_usage("generate")
def generate():
    quota_error = _consume_ai_quota("generate")
    if quota_error:
        return quota_error
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]
    gestures_raw = request.form.get("gestures", "[]")
    try:
        gestures = _validated_gestures(json.loads(gestures_raw))
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = os.path.join(tmp_dir, "recording.webm")
        audio_file.save(audio_path)

        try:
            transcript = _bounded_text(transcribe_audio(audio_path), "Transcript", MAX_TRANSCRIPT_CHARS, required=True)
            if moderate_text(transcript):
                return jsonify({"error": "This content can't be turned into a flowchart."}), 400

            flowchart_data = Flowchart.from_dict(extract_flowchart_json(transcript, gestures))
            title = generate_title(flowchart_data.to_dict())
            output_base = os.path.join(tmp_dir, "flowchart")
            pdf_path = render_flowchart(flowchart_data, output_base, title=title)
            return send_file(pdf_path, as_attachment=True, download_name=f"{sanitize_filename(title)}.pdf")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return _server_error("Could not generate the flowchart.")


@app.route("/live_update", methods=["POST"])
@login_required
@track_ai_usage("live_update")
def live_update():
    quota_error = _consume_ai_quota("live_update")
    if quota_error:
        return quota_error
    entitlement_error = _require_entitlement_available(_authenticated_client(), "generation")
    if entitlement_error:
        return entitlement_error
    data = request.get_json(silent=True) or {}
    try:
        transcript = _bounded_text(data.get("transcript"), "Transcript", MAX_TRANSCRIPT_CHARS)
        previous_graph = _validated_graph(data.get("previous_graph") or {"nodes": [], "edges": []})
        gestures = _validated_gestures(data.get("gestures"))
        updated = Flowchart.from_dict(update_graph_live(transcript, previous_graph, gestures))
        return jsonify(updated.to_dict())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return _server_error("Could not update the live flowchart.")


@app.route("/live_title", methods=["POST"])
@login_required
@track_ai_usage("live_title")
def live_title():
    quota_error = _consume_ai_quota("live_title")
    if quota_error:
        return quota_error
    entitlement_error = _require_entitlement_available(_authenticated_client(), "generation")
    if entitlement_error:
        return entitlement_error
    data = request.get_json(silent=True) or {}
    try:
        flowchart_data = _validated_graph(data.get("graph") or {"nodes": [], "edges": []})
        flowchart = Flowchart.from_dict(flowchart_data)
        return jsonify({"title": generate_title(flowchart.to_dict())})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return _server_error("Could not generate a flowchart title.")


@app.route("/finalize", methods=["POST"])
@login_required
@track_ai_usage("finalize")
def finalize_endpoint():
    quota_error = _consume_ai_quota("finalize")
    if quota_error:
        return quota_error
    data = request.get_json(silent=True) or {}
    try:
        flowchart_data = _validated_graph(data.get("graph") or {"nodes": [], "edges": []})
        transcript = _bounded_text(data.get("transcript"), "Transcript", MAX_TRANSCRIPT_CHARS)
        gestures = _validated_gestures(data.get("gestures"))
        if moderate_text(transcript):
            return jsonify({"error": "This content can't be turned into a flowchart."}), 400
        client = _authenticated_client()
        entitlement, entitlement_error = _reserve_entitlement(client, "generation")
        if entitlement_error:
            return entitlement_error
        reservation_id = entitlement.get("reservation_id")
        try:
            cleaned = build_layout_plan(Flowchart.from_dict(finalize_graph(flowchart_data, transcript, gestures)))
            _settle_entitlement(client, reservation_id)
            return jsonify({
                "graph": cleaned.to_dict(),
                "title": generate_title(cleaned.to_dict()),
                "generations_remaining": entitlement.get("remaining"),
            })
        except Exception:
            _refund_entitlement(client, reservation_id)
            raise
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return _server_error("Could not finalize the flowchart.")


@app.route("/ai_edit", methods=["POST"])
@login_required
@track_ai_usage("ai_edit")
def ai_edit():
    """Apply a focused natural-language edit to the current workflow."""
    quota_error = _consume_ai_quota("ai_edit")
    if quota_error:
        return quota_error
    data = request.get_json(silent=True) or {}
    try:
        graph = _validated_graph(data.get("graph") or {"nodes": [], "edges": []})
        instruction = _bounded_text(data.get("instruction"), "Edit instruction", MAX_INSTRUCTION_CHARS, required=True)
        started_at = time.perf_counter()
        if moderate_text(instruction):
            return jsonify({"error": "That edit instruction cannot be processed."}), 400
        client = _authenticated_client()
        entitlement, entitlement_error = _reserve_entitlement(client, "edit")
        if entitlement_error:
            return entitlement_error
        reservation_id = entitlement.get("reservation_id")
        try:
            edited = edit_flowchart_with_prompt(graph, instruction)
            _settle_entitlement(client, reservation_id)
            return jsonify({
                "graph": edited,
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
                "edits_remaining": entitlement.get("remaining"),
            })
        except Exception:
            _refund_entitlement(client, reservation_id)
            raise
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return _server_error("Could not apply the AI edit.")


@app.route("/export_pdf", methods=["POST"])
@login_required
@track_ai_usage("export")
def export_pdf():
    quota_error = _consume_ai_quota("export")
    if quota_error:
        return quota_error
    data = request.get_json(silent=True) or {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            flowchart_data = _validated_graph(data.get("graph") or {"nodes": [], "edges": []})
            transcript = _bounded_text(data.get("transcript"), "Transcript", MAX_TRANSCRIPT_CHARS)
            gestures = _validated_gestures(data.get("gestures"))
            if moderate_text(transcript):
                return jsonify({"error": "This content can't be turned into a flowchart."}), 400
            cleaned_data = flowchart_data if data.get("finalized") else finalize_graph(flowchart_data, transcript, gestures)
            cleaned = Flowchart.from_dict(cleaned_data)
            title = data.get("title") or generate_title(cleaned.to_dict())
            output_base = os.path.join(tmp_dir, "flowchart")
            pdf_path = render_flowchart(cleaned, output_base, title=title, start_node_id=cleaned.start_node_id)
            return send_file(pdf_path, as_attachment=True, download_name=f"{sanitize_filename(title)}.pdf")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return _server_error("Could not export the flowchart.")


@app.route("/export/<output_format>", methods=["POST"])
@login_required
@track_ai_usage("export")
def export_format(output_format):
    """Export the finalized canonical workflow as PDF, SVG, or PNG."""
    output_format = output_format.lower()
    if output_format not in SUPPORTED_FORMATS:
        return jsonify({"error": f"Unsupported export format: {output_format}"}), 400

    quota_error = _consume_ai_quota("export")
    if quota_error:
        return quota_error
    data = request.get_json(silent=True) or {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            flowchart_data = _validated_graph(data.get("graph") or {"nodes": [], "edges": []})
            transcript = _bounded_text(data.get("transcript"), "Transcript", MAX_TRANSCRIPT_CHARS)
            gestures = _validated_gestures(data.get("gestures"))
            if moderate_text(transcript):
                return jsonify({"error": "This content can't be turned into a flowchart."}), 400
            cleaned_data = flowchart_data if data.get("finalized") else finalize_graph(flowchart_data, transcript, gestures)
            cleaned = build_layout_plan(Flowchart.from_dict(cleaned_data))
            title = data.get("title") or generate_title(cleaned.to_dict())
            output_base = os.path.join(tmp_dir, "flowchart")
            output_path = render_flowchart(
                cleaned, output_base, title=title, start_node_id=cleaned.start_node_id, output_format=output_format
            )
            return send_file(
                output_path,
                as_attachment=True,
                download_name=f"{sanitize_filename(title)}.{output_format}",
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            return _server_error("Could not export the flowchart.")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
