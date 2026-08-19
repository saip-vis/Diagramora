"""Offline security and request-validation tests."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["SESSION_FILE_DIR"] = tempfile.mkdtemp(prefix="flowchart-test-sessions-")

from app import (
    MAX_INSTRUCTION_CHARS,
    _bounded_text,
    _clear_and_rotate_session,
    _estimated_cost_microusd,
    _refund_entitlement,
    _public_attempts,
    _public_rate_limited,
    _settle_entitlement,
    _validated_graph,
    app,
)
from flask import session
from ai_service import update_graph_live


class ValidationTests(unittest.TestCase):
    def test_entitlement_reservations_are_settled_or_refunded_by_id(self):
        client = Mock()
        client.rpc.return_value.execute.return_value = None

        _settle_entitlement(client, "settle-id")
        client.rpc.assert_called_with("settle_ai_entitlement", {"p_reservation_id": "settle-id"})

        _refund_entitlement(client, "refund-id")
        client.rpc.assert_called_with("refund_ai_entitlement", {"p_reservation_id": "refund-id"})

        call_count = client.rpc.call_count
        _settle_entitlement(client, None)
        _refund_entitlement(client, None)
        self.assertEqual(client.rpc.call_count, call_count)

    def test_schema_keeps_plan_and_refunds_out_of_broad_user_control(self):
        schema = Path(__file__).with_name("supabase_schema.sql").read_text()
        self.assertIn("grant update (username, display_name, updated_at)", schema)
        self.assertNotIn("grant select, insert, update, delete on public.profiles", schema)
        self.assertIn("refund_ai_entitlement(p_reservation_id uuid)", schema)
        self.assertIn("version_limit_reached", schema)
        self.assertIn("designs_workflow_size", schema)

    def test_cost_estimate_separates_cached_tokens(self):
        cost = _estimated_cost_microusd({
            "model": "gpt-4.1-nano-2025-04-14",
            "input_tokens": 1000,
            "cached_input_tokens": 400,
            "output_tokens": 200,
        })
        self.assertEqual(cost, 150)

    def test_unknown_model_cost_is_zero_but_can_still_be_logged(self):
        self.assertEqual(_estimated_cost_microusd({"model": "future-model", "input_tokens": 100}), 0)

    def test_bounded_text_rejects_oversized_input(self):
        with self.assertRaises(ValueError):
            _bounded_text("x" * (MAX_INSTRUCTION_CHARS + 1), "Instruction", MAX_INSTRUCTION_CHARS)

    def test_graph_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "unique ID"):
            _validated_graph({
                "nodes": [
                    {"id": "same", "label": "One"},
                    {"id": "same", "label": "Two"},
                ],
                "edges": [],
            })

    def test_graph_rejects_unknown_edge_endpoint(self):
        with self.assertRaisesRegex(ValueError, "existing nodes"):
            _validated_graph({
                "nodes": [{"id": "one", "label": "One"}],
                "edges": [{"from": "one", "to": "missing"}],
            })

    def test_valid_graph_is_normalized(self):
        graph = _validated_graph({
            "nodes": [{"id": "one", "label": "One"}, {"id": "two", "label": "Two"}],
            "edges": [{"from": "one", "to": "two", "label": "Next"}],
        })
        self.assertEqual(len(graph["nodes"]), 2)
        self.assertEqual(graph["edges"][0]["to"], "two")

    def test_live_update_drops_edges_to_undeclared_nodes(self):
        response = SimpleNamespace(
            model="gpt-4.1-nano",
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=(
                '{"new_nodes": [], "updated_nodes": [], '
                '"new_edges": [{"from": "one", "to": "node5", "label": ""}]}'
            )))],
        )
        with patch("ai_service.client.chat.completions.create", return_value=response):
            graph = update_graph_live(
                "Continue the process",
                {"nodes": [{"id": "one", "label": "Existing step"}], "edges": []},
                [],
            )
        self.assertEqual(graph["edges"], [])


class ApiBoundaryTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_ai_and_design_routes_require_login(self):
        for method, path in (
            ("post", "/live_update"),
            ("post", "/finalize"),
            ("post", "/ai_edit"),
            ("post", "/feedback"),
            ("get", "/designs"),
        ):
            response = getattr(self.client, method)(path, json={})
            self.assertEqual(response.status_code, 401, path)

    def test_cross_site_mutation_is_rejected(self):
        response = self.client.post(
            "/ai_edit",
            json={},
            headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
        )
        self.assertEqual(response.status_code, 403)

    def test_password_rules_are_enforced_before_supabase(self):
        response = self.client.post("/signup", json={
            "email": "person@example.com",
            "username": "person",
            "password": "short",
        })
        self.assertEqual(response.status_code, 400)

    def test_recovery_requires_tokens(self):
        response = self.client.post("/auth/complete-reset", json={"password": "long-enough-password"})
        self.assertEqual(response.status_code, 400)

    def test_security_headers_are_present(self):
        response = self.client.get("/")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
        self.assertIn("camera=(self)", response.headers["Permissions-Policy"])

    def test_auth_boundary_rotates_server_session_id(self):
        with app.test_request_context("/"):
            session["supabase_access_token"] = "old-user-token"
            old_session_id = session.sid
            _clear_and_rotate_session()
            self.assertNotEqual(session.sid, old_session_id)
            self.assertNotIn("supabase_access_token", session)

    def test_successful_login_checks_do_not_consume_failure_budget(self):
        _public_attempts.clear()
        with app.test_request_context("/login", environ_base={"REMOTE_ADDR": "127.0.0.9"}):
            for _ in range(20):
                self.assertFalse(_public_rate_limited("login", 12, 900, record_attempt=False))
            for _ in range(12):
                self.assertFalse(_public_rate_limited("login", 12, 900))
            self.assertTrue(_public_rate_limited("login", 12, 900, record_attempt=False))

    def test_public_metadata_and_error_routes(self):
        for path, expected_type in (
            ("/favicon.svg", "image/svg+xml"),
            ("/robots.txt", "text/plain"),
            ("/sitemap.xml", "application/xml"),
        ):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn(expected_type, response.content_type)
            response.close()
        self.assertEqual(self.client.get("/missing-page").status_code, 404)


if __name__ == "__main__":
    unittest.main()
