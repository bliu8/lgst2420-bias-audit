"""Unit tests for audit/probes.py — no I/O except fixture CSV reads."""
import pytest
from audit.probes import render_probes, NAMES, _normalize


class TestRenderProbes:
    def test_returns_list(self):
        probes = render_probes("hiring")
        assert isinstance(probes, list)
        assert len(probes) > 0

    def test_all_categories(self):
        for cat in ["hiring", "lending", "medical"]:
            probes = render_probes(cat)
            assert len(probes) > 0, f"No probes for category {cat}"

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="Unknown category"):
            render_probes("nonexistent_category")

    def test_probe_fields(self):
        probes = render_probes("hiring")
        probe = probes[0]
        assert probe.probe_id
        assert probe.category == "hiring"
        assert probe.group
        assert probe.subgroup
        assert probe.text

    def test_no_unfilled_placeholders(self):
        probes = render_probes("hiring")
        for probe in probes:
            assert "{name}" not in probe.text, (
                f"Unfilled {{name}} in probe {probe.probe_id}"
            )
            assert "{" not in probe.text or "}" not in probe.text or \
                   not any(
                       probe.text[i] == "{" and probe.text.find("}", i) > 0
                       for i in range(len(probe.text))
                       if probe.text[i] == "{"
                   ), f"Possible unfilled placeholder in: {probe.text}"

    def test_variant_count(self):
        probes = render_probes("hiring")
        total_names = sum(len(v) for v in NAMES.values())
        # 1 template × total_names names = total_names probes
        assert len(probes) == total_names

    def test_no_cross_contamination(self):
        probes = render_probes("hiring")
        white_male_names = set(NAMES["white_male"])
        black_female_names = set(NAMES["black_female"])
        for probe in probes:
            if probe.subgroup == "white_male":
                assert not any(n in probe.text for n in black_female_names), (
                    f"Black female name in white_male probe: {probe.text}"
                )
            if probe.subgroup == "black_female":
                assert not any(n in probe.text for n in white_male_names), (
                    f"White male name in black_female probe: {probe.text}"
                )

    def test_subgroup_coverage(self):
        probes = render_probes("hiring")
        subgroups = {p.subgroup for p in probes}
        assert subgroups == set(NAMES.keys())

    def test_unique_probe_ids(self):
        probes = render_probes("hiring")
        ids = [p.probe_id for p in probes]
        assert len(ids) == len(set(ids))

    def test_name_group_filter(self):
        probes = render_probes("hiring", name_groups=["white_male"])
        assert all(p.subgroup == "white_male" for p in probes)
        assert len(probes) == len(NAMES["white_male"])

    def test_from_csv(self, sample_probes_csv):
        probes = render_probes("hiring", template_csv=sample_probes_csv)
        assert len(probes) > 0
        for probe in probes:
            assert "{name}" not in probe.text

    def test_whitespace_normalization(self):
        probes = render_probes("hiring")
        for probe in probes:
            assert "  " not in probe.text
            assert probe.text == probe.text.strip()


class TestNormalize:
    def test_collapses_spaces(self):
        assert _normalize("hello   world") == "hello world"

    def test_strips_edges(self):
        assert _normalize("  hello  ") == "hello"

    def test_no_op_on_clean(self):
        assert _normalize("hello world") == "hello world"
