"""
Tests for audit/runner.py.

All tests use a mocked Anthropic client — no live API calls.
Live integration tests are marked with @pytest.mark.integration.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from audit.probes import render_probes, Probe
from audit.runner import AuditRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(tmp_path: Path) -> AuditRunner:
    runner = AuditRunner(
        model="claude-haiku-4-5-20251001",
        api_key="test-key",
        results_dir=tmp_path,
    )
    return runner


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    msg.content = [content_block]
    return msg


# ---------------------------------------------------------------------------
# AuditRunner.run — happy path
# ---------------------------------------------------------------------------

class TestRunnerHappyPath:
    def test_result_count_matches_probes(self, tmp_path):
        probes = render_probes("hiring")[:4]
        runner = _make_runner(tmp_path)
        with patch.object(runner.client.messages, "create",
                          return_value=_mock_response("Yes, recommend.")):
            results = runner.run(probes)
        assert len(results) == len(probes)

    def test_result_structure(self, tmp_path):
        probes = render_probes("hiring")[:1]
        runner = _make_runner(tmp_path)
        with patch.object(runner.client.messages, "create",
                          return_value=_mock_response("Yes.")):
            results = runner.run(probes)
        r = results[0]
        required = {"probe_id", "category", "group", "subgroup",
                    "prompt", "completion", "latency_ms", "error"}
        assert required.issubset(r.keys())

    def test_completion_recorded(self, tmp_path):
        probes = render_probes("hiring")[:1]
        runner = _make_runner(tmp_path)
        with patch.object(runner.client.messages, "create",
                          return_value=_mock_response("Absolutely yes!")):
            results = runner.run(probes)
        assert results[0]["completion"] == "Absolutely yes!"

    def test_no_error_on_success(self, tmp_path):
        probes = render_probes("hiring")[:1]
        runner = _make_runner(tmp_path)
        with patch.object(runner.client.messages, "create",
                          return_value=_mock_response("Yes.")):
            results = runner.run(probes)
        assert results[0]["error"] is None

    def test_latency_recorded(self, tmp_path):
        probes = render_probes("hiring")[:1]
        runner = _make_runner(tmp_path)
        with patch.object(runner.client.messages, "create",
                          return_value=_mock_response("Yes.")):
            results = runner.run(probes)
        assert isinstance(results[0]["latency_ms"], int)
        assert results[0]["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_retries_on_rate_limit(self, tmp_path):
        import anthropic
        probes = render_probes("hiring")[:1]
        runner = _make_runner(tmp_path)

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise anthropic.RateLimitError(
                    "rate limit", response=MagicMock(), body={}
                )
            return _mock_response("Yes, eventually.")

        with patch.object(runner.client.messages, "create", side_effect=side_effect):
            with patch("time.sleep"):  # suppress actual sleep
                results = runner.run(probes, max_retries=3, delay=0.0)

        assert results[0]["error"] is None
        assert results[0]["completion"] == "Yes, eventually."
        assert call_count == 3

    def test_permanent_failure_marked_as_error(self, tmp_path):
        import anthropic
        probes = render_probes("hiring")[:1]
        runner = _make_runner(tmp_path)

        def always_fail(*args, **kwargs):
            raise anthropic.RateLimitError(
                "rate limit", response=MagicMock(), body={}
            )

        with patch.object(runner.client.messages, "create", side_effect=always_fail):
            with patch("time.sleep"):
                results = runner.run(probes, max_retries=2, delay=0.0)

        assert results[0]["error"] is not None
        assert results[0]["completion"] == ""

    def test_partial_failure_does_not_stop_run(self, tmp_path):
        import anthropic
        probes = render_probes("hiring")[:3]
        runner = _make_runner(tmp_path)

        call_count = 0
        def mixed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise anthropic.APIError("oops", request=MagicMock(), body={})
            return _mock_response("Yes.")

        with patch.object(runner.client.messages, "create", side_effect=mixed):
            with patch("time.sleep"):
                results = runner.run(probes, max_retries=1, delay=0.0)

        assert len(results) == 3
        errors = [r for r in results if r["error"]]
        successes = [r for r in results if not r["error"]]
        assert len(errors) == 1
        assert len(successes) == 2


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

class TestRunnerSave:
    def test_saves_json(self, tmp_path):
        runner = _make_runner(tmp_path)
        results = [{"probe_id": "x", "completion": "yes", "error": None}]
        out = runner.save(results, tag="hiring")
        assert out.exists()
        with open(out) as f:
            loaded = json.load(f)
        assert loaded == results

    def test_latest_json_written(self, tmp_path):
        runner = _make_runner(tmp_path)
        results = [{"probe_id": "x", "completion": "yes", "error": None}]
        runner.save(results)
        assert (tmp_path / "latest.json").exists()

    def test_results_dir_created(self, tmp_path):
        subdir = tmp_path / "nested" / "results"
        runner = AuditRunner("model", "key", subdir)
        assert subdir.exists()


# ---------------------------------------------------------------------------
# Integration (live API)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRunnerIntegration:
    def test_live_small_run(self, tmp_path):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        probes = render_probes("hiring")[:5]
        runner = AuditRunner(
            model="claude-haiku-4-5-20251001",
            api_key=api_key,
            results_dir=tmp_path,
        )
        results = runner.run(probes)
        assert len(results) == 5

    def test_live_latency_reasonable(self, tmp_path):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        probes = render_probes("hiring")[:2]
        runner = AuditRunner(
            model="claude-haiku-4-5-20251001",
            api_key=api_key,
            results_dir=tmp_path,
        )
        results = runner.run(probes)
        for r in results:
            if not r["error"]:
                assert r["latency_ms"] < 30_000, "Single call took > 30s"
