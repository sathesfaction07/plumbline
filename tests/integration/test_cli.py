"""CLI: every command, exit codes, and stdout staying pipeable."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from ai_classifier.cli import main
from tests.conftest import DATA

PRODUCTS = str(DATA / "products.csv")
LABELS = str(DATA / "labels.csv")


@pytest.fixture
def run():
    runner = CliRunner()
    return lambda *args: runner.invoke(main, list(args))


class TestMeasure:
    def test_it_reports_and_exits_nonzero_on_a_blind_spot(self, run):
        """The offline provider cannot see images, so image-only is unmeasurable."""
        result = run("measure", "-i", PRODUCTS, "-l", LABELS, "--no-confusion")
        assert result.exit_code == 1
        assert "blind spots" in result.output
        assert "image_only" in result.output
        assert "NONE answered" in result.output

    def test_the_modality_table_shows_n_a_not_a_zero_score(self, run):
        result = run("measure", "-i", PRODUCTS, "-l", LABELS, "--no-confusion")
        row = next(line for line in result.output.splitlines() if "image_only" in line
                   and "%" in line)
        assert "n/a" in row
        assert "0.0000" not in row

    def test_the_confusion_matrix_is_shown_by_default_and_suppressible(self, run):
        with_it = run("measure", "-i", PRODUCTS, "-l", LABELS)
        without = run("measure", "-i", PRODUCTS, "-l", LABELS, "--no-confusion")
        assert "confusion" in with_it.output
        assert "confusion" not in without.output

    def test_equal_count_binning_is_selectable(self, run):
        result = run("measure", "-i", PRODUCTS, "-l", LABELS, "-s", "equal_count",
                     "--no-confusion")
        assert "equal_count bins" in result.output

    def test_an_export_writes_all_four_artefacts(self, run, tmp_path):
        result = run("measure", "-i", PRODUCTS, "-l", LABELS, "--no-confusion",
                     "-o", str(tmp_path))
        assert result.exit_code == 1, "still non-zero: the blind spot has not gone away"
        names = {path.name for path in tmp_path.iterdir()}
        assert names == {"outcomes.csv", "bins.csv", "outcomes.jsonl", "report.json"}

    def test_the_json_report_carries_null_for_an_unmeasurable_subgroup(self, run, tmp_path):
        run("measure", "-i", PRODUCTS, "-l", LABELS, "--no-confusion", "-o", str(tmp_path))
        payload = json.loads((tmp_path / "report.json").read_text())
        assert payload["by_modality"]["image_only"]["ece"] is None
        assert payload["by_modality"]["image_only"]["measurable"] is False
        assert payload["by_modality"]["both"]["ece"] is not None
        assert [item["subgroup"] for item in payload["blind_spots"]] == [
            "image_only", "neither"
        ]

    def test_undecided_labels_are_reported_on_stderr_and_excluded(self, run, tmp_path):
        run("measure", "-i", PRODUCTS, "-l", LABELS, "--no-confusion", "-o", str(tmp_path))
        payload = json.loads((tmp_path / "report.json").read_text())
        assert payload["summary"]["items"] == 280, "19 undecided labels are excluded"

    def test_an_unreachable_target_is_stated_as_advice(self, run):
        result = run("measure", "-i", PRODUCTS, "-l", LABELS, "--no-confusion", "-t", "0.99")
        assert "NOT REACHABLE" in result.output
        assert result.exit_code == 1

    def test_a_reachable_target_still_fails_on_the_blind_spot(self, run):
        """Two independent reasons to exit non-zero; neither masks the other."""
        result = run("measure", "-i", PRODUCTS, "-l", LABELS, "--no-confusion", "-t", "0.5")
        assert "auto-accept at confidence" in result.output
        assert result.exit_code == 1, "the blind spot alone is enough"


class TestClassify:
    def test_it_writes_jsonl_and_exits_zero(self, run, tmp_path):
        out = tmp_path / "c.jsonl"
        result = run("classify", "-i", PRODUCTS, "-o", str(out))
        assert result.exit_code == 0
        lines = [line for line in out.read_text().splitlines() if line.strip()]
        assert len(lines) == 340
        for line in lines[:5]:
            json.loads(line)

    def test_a_second_run_resumes_from_the_checkpoint(self, run, tmp_path):
        checkpoint = tmp_path / "ck.jsonl"
        first = run("classify", "-i", PRODUCTS, "-c", str(checkpoint))
        assert "newly run       340" in first.output
        second = run("classify", "-i", PRODUCTS, "-c", str(checkpoint))
        assert "from checkpoint 340" in second.output
        assert "newly run       0" in second.output

    def test_an_unknown_provider_is_rejected_by_click(self, run):
        result = run("classify", "-i", PRODUCTS, "-p", "bedroc")
        assert result.exit_code == 2


class TestStability:
    def test_it_shows_the_recommendation_over_growing_label_counts(self, run):
        result = run("stability", "-i", PRODUCTS, "-l", LABELS, "--steps", "4")
        assert "labels used" in result.output
        assert result.output.count("\n") >= 5

    def test_an_unreachable_target_shows_none_reachable(self, run):
        result = run("stability", "-i", PRODUCTS, "-l", LABELS, "-t", "0.99", "--steps", "3")
        assert "none reachable" in result.output


class TestProvidersCommand:
    def test_it_lists_all_three_and_explains_the_blind_spot(self, run):
        result = run("providers")
        assert result.exit_code == 0
        for name in ("offline", "bedrock", "vertex"):
            assert name in result.output
        assert "abstains on every image-only listing" in result.output


class TestDoctor:
    def test_it_passes_with_the_shipped_data(self, run):
        result = run("doctor", "-i", PRODUCTS, "-l", LABELS)
        assert result.exit_code == 0
        assert "340 row(s)" in result.output
        assert "19 undecided" in result.output

    def test_it_explains_why_n_a_is_printed_instead_of_zero(self, run):
        result = run("doctor", "-i", PRODUCTS, "-l", LABELS)
        assert "0.0000 is the BEST possible ECE" in result.output

    def test_a_missing_input_fails(self, run, tmp_path):
        result = run("doctor", "-i", str(tmp_path / "nope.csv"), "-l", LABELS)
        assert result.exit_code == 1
        assert "MISSING" in result.output


class TestStdoutStaysPipeable:
    def test_diagnostics_go_to_stderr_not_stdout(self):
        """`measure | grep` must not pick up warnings about unlabelled products.

        Click 8.2 separated the streams by default and dropped the `mix_stderr` argument,
        so no flag is needed -- `result.stdout` and `result.stderr` are already distinct.
        """
        runner = CliRunner()
        result = runner.invoke(
            main, ["measure", "-i", PRODUCTS, "-l", LABELS, "--no-confusion"]
        )
        assert "have no label" in result.stderr
        assert "have no label" not in result.stdout
        assert "summary" in result.stdout
