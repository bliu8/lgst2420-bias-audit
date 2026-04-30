"""Unit tests for audit/metrics.py — pure functions, no I/O."""
import pytest
from audit.metrics import (
    positive_rate,
    refusal_rate,
    sentiment_score,
    demographic_parity_gap,
    aggregate_disparity,
    summarize_results,
)


# ---------------------------------------------------------------------------
# positive_rate
# ---------------------------------------------------------------------------

class TestPositiveRate:
    def test_all_positive(self):
        completions = ["Yes, I recommend.", "Absolutely yes, advance.", "Yes move forward."]
        assert positive_rate(completions, ["yes"]) == pytest.approx(1.0)

    def test_none_positive(self):
        completions = ["No, do not proceed.", "Decline this one.", "Pass."]
        assert positive_rate(completions, ["yes", "recommend"]) == pytest.approx(0.0)

    def test_mixed(self):
        completions = ["Yes.", "No.", "Yes.", "No."]
        assert positive_rate(completions, ["yes"]) == pytest.approx(0.5)

    def test_case_insensitive(self):
        completions = ["YES, ADVANCE.", "yes please.", "No."]
        assert positive_rate(completions, ["yes"]) == pytest.approx(2 / 3)

    def test_empty_completions(self):
        assert positive_rate([], ["yes"]) is None

    def test_single_item_positive(self):
        assert positive_rate(["yes"], ["yes"]) == pytest.approx(1.0)

    def test_single_item_negative(self):
        assert positive_rate(["no"], ["yes"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# refusal_rate
# ---------------------------------------------------------------------------

class TestRefusalRate:
    REFUSALS = ["I cannot", "I can't", "I won't"]

    def test_all_refusal(self):
        completions = ["I cannot answer this.", "I can't do that.", "I won't respond."]
        assert refusal_rate(completions, self.REFUSALS) == pytest.approx(1.0)

    def test_no_refusal(self):
        completions = ["Yes.", "No.", "Sure."]
        assert refusal_rate(completions, self.REFUSALS) == pytest.approx(0.0)

    def test_mixed(self):
        completions = ["I cannot help.", "Yes, proceed.", "I won't do that.", "No."]
        assert refusal_rate(completions, self.REFUSALS) == pytest.approx(0.5)

    def test_empty(self):
        assert refusal_rate([], self.REFUSALS) is None

    def test_single_refusal(self):
        assert refusal_rate(["I cannot"], self.REFUSALS) == pytest.approx(1.0)

    def test_single_non_refusal(self):
        assert refusal_rate(["Yes."], self.REFUSALS) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# sentiment_score
# ---------------------------------------------------------------------------

class TestSentimentScore:
    def test_clearly_positive(self):
        score = sentiment_score("Yes, I recommend advancing this excellent candidate.")
        assert score > 0

    def test_clearly_negative(self):
        score = sentiment_score("No, I would deny and reject this poor application.")
        assert score < 0

    def test_neutral(self):
        score = sentiment_score("The candidate submitted their application today.")
        assert score == pytest.approx(0.0)

    def test_bounds(self):
        score = sentiment_score("yes yes yes yes yes")
        assert -1.0 <= score <= 1.0

    def test_empty_string(self):
        assert sentiment_score("") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# demographic_parity_gap
# ---------------------------------------------------------------------------

class TestDemographicParityGap:
    def test_zero_gap(self):
        rates = {"A": 0.8, "B": 0.8, "C": 0.8}
        assert demographic_parity_gap(rates) == pytest.approx(0.0)

    def test_max_gap(self):
        rates = {"A": 1.0, "B": 0.0}
        assert demographic_parity_gap(rates) == pytest.approx(1.0)

    def test_fractional(self):
        rates = {"A": 0.9, "B": 0.6, "C": 0.75}
        assert demographic_parity_gap(rates) == pytest.approx(0.3)

    def test_single_group(self):
        assert demographic_parity_gap({"A": 0.5}) is None

    def test_none_value_excluded(self):
        # If any value is None, return None
        assert demographic_parity_gap({"A": 0.8, "B": None}) is None

    def test_all_none(self):
        assert demographic_parity_gap({"A": None, "B": None}) is None


# ---------------------------------------------------------------------------
# aggregate_disparity
# ---------------------------------------------------------------------------

class TestAggregateDisparity:
    def test_equal_weights(self):
        gaps = {"pr": 0.4, "rr": 0.2}
        result = aggregate_disparity(gaps)
        assert result == pytest.approx(0.3)

    def test_custom_weights(self):
        gaps = {"pr": 0.4, "rr": 0.0}
        result = aggregate_disparity(gaps, weights={"pr": 2.0, "rr": 1.0})
        assert result == pytest.approx((0.4 * 2 + 0.0 * 1) / 3)

    def test_all_none(self):
        assert aggregate_disparity({"pr": None, "rr": None}) is None

    def test_some_none_skipped(self):
        gaps = {"pr": 0.6, "rr": None}
        result = aggregate_disparity(gaps)
        assert result == pytest.approx(0.6)

    def test_zero_gaps(self):
        assert aggregate_disparity({"pr": 0.0, "rr": 0.0}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# summarize_results (integration of metrics)
# ---------------------------------------------------------------------------

class TestSummarizeResults:
    def test_basic_structure(self, canned_results):
        summary = summarize_results(canned_results, positive_keywords=["yes", "recommend"])
        assert "subgroups" in summary
        assert "gaps" in summary
        assert "aggregate_disparity" in summary

    def test_subgroup_keys(self, canned_results):
        summary = summarize_results(canned_results, positive_keywords=["yes", "recommend"])
        subgroups = set(summary["subgroups"].keys())
        assert "white_male" in subgroups
        assert "black_male" in subgroups

    def test_positive_rate_gap_direction(self, canned_results):
        summary = summarize_results(canned_results, positive_keywords=["yes", "recommend"])
        # Fixture has white > black positive rates
        sg = summary["subgroups"]
        assert sg["white_male"]["positive_rate"] > sg["black_male"]["positive_rate"]

    def test_gap_non_negative(self, canned_results):
        summary = summarize_results(canned_results, positive_keywords=["yes", "recommend"])
        gap = summary["gaps"]["positive_rate_gap"]
        assert gap is None or gap >= 0

    def test_errors_excluded(self):
        results = [
            {"probe_id": "x", "category": "hiring", "group": "race_gender",
             "subgroup": "white_male", "prompt": "p", "completion": "yes", "error": None},
            {"probe_id": "y", "category": "hiring", "group": "race_gender",
             "subgroup": "white_male", "prompt": "p", "completion": "",  "error": "rate_limit"},
        ]
        summary = summarize_results(results, positive_keywords=["yes"])
        assert summary["subgroups"]["white_male"]["n"] == 1

    def test_empty_results(self):
        summary = summarize_results([])
        assert summary["subgroups"] == {}
        assert summary["aggregate_disparity"] is None
