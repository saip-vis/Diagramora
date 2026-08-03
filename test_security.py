"""Offline security and request-validation tests."""

import os
import tempfile
import unittest

os.environ["SESSION_FILE_DIR"] = tempfile.mkdtemp(prefix="flowchart-test-sessions-")

from app import (
    MAX_INSTRUCTION_CHARS,
    _bounded_text,
    _validated_graph,
    app,
)


class ValidationTests(unittest.TestCase):
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


class ApiBoundaryTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_ai_and_design_routes_require_login(self):
        for method, path in (
            ("post", "/live_update"),
            ("post", "/finalize"),
            ("post", "/ai_edit"),
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


if __name__ == "__main__":
    unittest.main()
