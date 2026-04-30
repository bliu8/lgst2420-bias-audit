"""Unit tests for audit/report.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.report import generate_report, _render_markdown, _MD_REQUIRED_SECTIONS, _REPORT_SCHEMA_KEYS


class TestGenerateReport:
    def test_creates_both_files(self, sample_results_path, tmp_path):
        md, js = generate_report(sample_results_path, output_dir=tmp_path)
        assert md.exists()
        assert js.exists()

    def test_json_schema(self, sample_results_path, tmp_path):
        _, js = generate_report(sample_results_path, output_dir=tmp_path)
        with open(js) as f:
            data = json.load(f)
        assert _REPORT_SCHEMA_KEYS.issubset(data.keys()), (
            f"Missing keys: {_REPORT_SCHEMA_KEYS - data.keys()}"
        )

    def test_round_trip_accuracy(self, sample_results_path, tmp_path):
        """Numbers in the JSON report must match what summarize_results produces."""
        from audit.metrics import summarize_results
        from audit.probes import POSITIVE_KEYWORDS, REFUSAL_STRINGS
        import json as _json

        with open(sample_results_path) as f:
            results = _json.load(f)
        category = results[0].get("category", "hiring")
        expected = summarize_results(
            results,
            positive_keywords=POSITIVE_KEYWORDS.get(category, ["yes"]),
            refusal_strings=REFUSAL_STRINGS,
        )

        _, js = generate_report(sample_results_path, output_dir=tmp_path)
        with open(js) as f:
            report = _json.load(f)

        # Check aggregate disparity matches to 4 decimal places
        exp_agg = expected["aggregate_disparity"]
        rep_agg = report["aggregate_disparity"]
        if exp_agg is not None and rep_agg is not None:
            assert abs(exp_agg - rep_agg) < 1e-4

        # Check per-subgroup positive rates
        for sg, stats in expected["subgroups"].items():
            if stats["positive_rate"] is not None and sg in report["subgroups"]:
                assert abs(stats["positive_rate"] - report["subgroups"][sg]["positive_rate"]) < 1e-4

    def test_markdown_required_sections(self, sample_results_path, tmp_path):
        md, _ = generate_report(sample_results_path, output_dir=tmp_path)
        content = md.read_text()
        for section in _MD_REQUIRED_SECTIONS:
            assert section in content, f"Missing section: {section}"

    def test_threshold_flagging(self, sample_results_path, tmp_path):
        # Fixture has white groups at 100% positive, black groups much lower → lots of flagged
        _, js = generate_report(sample_results_path, output_dir=tmp_path, threshold=0.05)
        with open(js) as f:
            data = json.load(f)
        assert len(data["flagged"]) > 0

    def test_high_threshold_no_flagged(self, sample_results_path, tmp_path):
        _, js = generate_report(sample_results_path, output_dir=tmp_path, threshold=0.99)
        with open(js) as f:
            data = json.load(f)
        assert data["flagged"] == []

    def test_empty_results(self, tmp_path):
        empty_path = tmp_path / "empty.json"
        empty_path.write_text("[]")
        md, js = generate_report(empty_path, output_dir=tmp_path)
        assert md.exists()
        with open(js) as f:
            data = json.load(f)
        assert data["n_total"] == 0
        assert data["flagged"] == []

    def test_n_total_correct(self, sample_results_path, tmp_path):
        import json as _json
        with open(sample_results_path) as f:
            raw = _json.load(f)
        _, js = generate_report(sample_results_path, output_dir=tmp_path)
        with open(js) as f:
            data = _json.load(f)
        assert data["n_total"] == len(raw)

    def test_n_errors_correct(self, tmp_path):
        results = [
            {"probe_id": "a", "category": "hiring", "group": "g", "subgroup": "white_male",
             "prompt": "p", "completion": "yes", "error": None},
            {"probe_id": "b", "category": "hiring", "group": "g", "subgroup": "black_male",
             "prompt": "p", "completion": "", "error": "rate_limit"},
        ]
        rpath = tmp_path / "test.json"
        rpath.write_text(json.dumps(results))
        _, js = generate_report(rpath, output_dir=tmp_path)
        with open(js) as f:
            data = json.load(f)
        assert data["n_errors"] == 1


class TestRenderMarkdown:
    """Test _render_markdown directly with synthetic data."""

    SAMPLE_DATA = {
        "generated_at": "2026-01-01T00:00:00",
        "source_file": "test.json",
        "category": "hiring",
        "n_total": 20,
        "n_errors": 0,
        "subgroups": {
            "white_male":   {"n": 5, "positive_rate": 1.0,  "refusal_rate": 0.0, "mean_sentiment": 1.0},
            "black_male":   {"n": 5, "positive_rate": 0.4,  "refusal_rate": 0.0, "mean_sentiment": -0.5},
        },
        "gaps": {
            "positive_rate_gap": 0.6,
            "refusal_rate_gap": 0.0,
            "sentiment_gap": 1.5,
        },
        "aggregate_disparity": 0.7,
        "flagged": [
            {"probe_id": "x", "subgroup": "black_male",
             "prompt_excerpt": "Candidate Jamal…", "completion": "No."},
        ],
    }

    def test_all_required_sections_present(self):
        md = _render_markdown(self.SAMPLE_DATA, threshold=0.1)
        for section in _MD_REQUIRED_SECTIONS:
            assert section in md

    def test_aggregate_score_appears(self):
        md = _render_markdown(self.SAMPLE_DATA, threshold=0.1)
        assert "70.0%" in md  # 0.7 as percentage

    def test_flagged_completion_appears(self):
        md = _render_markdown(self.SAMPLE_DATA, threshold=0.1)
        assert "Jamal" in md

    def test_no_flagged_shows_message(self):
        data = {**self.SAMPLE_DATA, "flagged": []}
        md = _render_markdown(data, threshold=0.1)
        assert "No completions flagged" in md
