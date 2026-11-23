"""Rendering and export: the presentation layer must not restate the metric bug."""

from __future__ import annotations

import csv
import json

from ai_classifier.core.calibrate import (
    calibrate,
    calibrate_by_modality,
    choose_threshold,
)
from ai_classifier.models import Modality
from ai_classifier.render import export, text
from tests.conftest import abstentions, many, requires_matplotlib


class TestUnmeasurableRendersAsNotAvailable:
    """The regression, at the presentation layer.

    `0.0000` in an ECE column is the best possible score. A renderer printing it for
    "nothing was answered" restates the exact inversion the model layer now prevents.
    """

    def test_the_summary_prints_n_a_not_zero(self):
        block = "\n".join(text.summary_block(calibrate(abstentions(40))))
        assert "n/a" in block
        assert "0.0000" not in block
        assert "would read as a perfect result" in block

    def test_a_measured_summary_prints_real_numbers(self):
        block = "\n".join(text.summary_block(calibrate(many(40, confidence=0.9, accuracy=0.7))))
        assert "n/a" not in block
        assert "0.2" in block

    def test_the_modality_table_distinguishes_blind_from_measured(self):
        got = many(40, confidence=0.9, accuracy=0.7, modality=Modality.BOTH)
        got += abstentions(20, modality=Modality.IMAGE_ONLY)
        rows = text.modality_table(calibrate_by_modality(got))
        blind = next(row for row in rows if "image_only" in row)
        seeing = next(row for row in rows if "both" in row)
        assert "n/a" in blind
        assert "n/a" not in seeing
        assert blind.split() != seeing.split()

    def test_an_empty_reliability_curve_says_why(self):
        lines = text.reliability_diagram(calibrate(abstentions(10)))
        assert any("nothing was answered" in line for line in lines)

    def test_blind_spots_are_named_in_the_report(self):
        got = many(40, confidence=0.9, accuracy=0.7, modality=Modality.BOTH)
        got += abstentions(43, modality=Modality.IMAGE_ONLY)
        report = calibrate(got)
        rendered = text.render(report, got, choose_threshold(got))
        assert "blind spots" in rendered
        assert "image_only" in rendered
        assert "NONE answered" in rendered

    def test_no_blind_spots_says_so_explicitly(self):
        got = many(40, confidence=0.9, accuracy=0.7, modality=Modality.BOTH)
        rendered = text.render(report := calibrate(got), got, choose_threshold(got))
        assert report.measurable
        assert "every modality had enough answers" in rendered


class TestReliabilityDiagram:
    def test_one_row_per_populated_bin_plus_header_and_legend(self):
        got = many(20, confidence=0.95, accuracy=0.8) + many(
            20, confidence=0.65, accuracy=0.6, prefix="Q"
        )
        lines = text.reliability_diagram(calibrate(got))
        assert len(lines) == 2 + 2

    def test_a_thin_bin_is_marked(self):
        got = many(30, confidence=0.95, accuracy=0.8) + many(
            3, confidence=0.15, accuracy=0.3, prefix="Q"
        )
        lines = text.reliability_diagram(calibrate(got))
        assert any("(thin)" in line for line in lines)

    def test_claimed_and_observed_marks_both_appear(self):
        lines = text.reliability_diagram(calibrate(many(20, confidence=0.95, accuracy=0.55)))
        row = lines[1]
        assert "|" in row and "#" in row

    def test_a_perfectly_calibrated_bin_shows_the_observed_mark(self):
        """They coincide; observed must win the cell rather than being hidden."""
        lines = text.reliability_diagram(calibrate(many(20, confidence=0.9, accuracy=0.9)))
        assert "#" in lines[1]


class TestThresholdRendering:
    def test_an_unreachable_target_is_stated_as_advice_not_as_a_number(self):
        got = many(40, confidence=0.95, accuracy=0.5)
        block = "\n".join(text.threshold_block(choose_threshold(got, target_accuracy=0.95)))
        assert "NOT REACHABLE" in block
        assert "route all 40 to human review" in block
        assert ">= 1.0" not in block, "a threshold of 1.0 would be a rule that fires"

    def test_a_reachable_target_shows_the_cut_off(self):
        got = many(40, confidence=0.99, accuracy=1.0)
        block = "\n".join(text.threshold_block(choose_threshold(got, target_accuracy=0.95)))
        assert ">= 0.99" in block
        assert "automates 100.0%" in block


class TestExport:
    def test_an_unmeasurable_value_serialises_as_null_not_zero(self, tmp_path):
        got = abstentions(30, modality=Modality.IMAGE_ONLY)
        payload = export.report_payload(calibrate(got), got, choose_threshold(got))
        assert payload["summary"]["ece"] is None
        assert payload["summary"]["accuracy"] is None
        assert payload["summary"]["measurable"] is False
        raw = json.dumps(payload)
        assert '"ece": null' in raw

    def test_a_blind_spot_appears_in_the_json_report(self):
        got = many(30, confidence=0.9, accuracy=0.8, modality=Modality.BOTH)
        got += abstentions(43, modality=Modality.IMAGE_ONLY)
        payload = export.report_payload(calibrate(got), got, choose_threshold(got))
        assert [item["subgroup"] for item in payload["blind_spots"]] == ["image_only"]

    def test_an_abstention_exports_as_abstained_not_as_blank(self, tmp_path):
        """Blank reads as missing data; declining is a result."""
        rows = export.outcome_rows(abstentions(3))
        assert {row["correct"] for row in rows} == {"abstained"}

    def test_all_files_are_written_even_when_empty(self, tmp_path):
        report = calibrate([])
        written = export.export_all(report, [], choose_threshold([]), tmp_path)
        names = {path.name for path in written}
        assert names == {"outcomes.csv", "bins.csv", "outcomes.jsonl", "report.json"}
        for path in written:
            assert path.exists()

    def test_an_empty_csv_still_carries_its_header(self, tmp_path):
        export.export_all(calibrate([]), [], choose_threshold([]), tmp_path)
        rows = list(csv.reader((tmp_path / "outcomes.csv").read_text().splitlines()))
        assert rows[0] == export.OUTCOME_FIELDS
        assert len(rows) == 1

    def test_jsonl_is_one_valid_object_per_line(self, tmp_path):
        got = many(5, confidence=0.9, accuracy=0.8)
        path = export.write_jsonl(export.outcome_rows(got), tmp_path / "o.jsonl")
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 5
        for line in lines:
            json.loads(line)

    def test_bins_export_carries_the_subgroup(self, tmp_path):
        got = many(20, confidence=0.9, accuracy=0.8, modality=Modality.BOTH)
        got += many(20, confidence=0.7, accuracy=0.6, modality=Modality.TEXT_ONLY, prefix="T")
        rows = export.bin_rows(calibrate_by_modality(got))
        assert {row["subgroup"] for row in rows} == {"both", "text_only"}

    def test_the_export_uses_the_requested_bin_count_not_the_populated_one(self, tmp_path):
        got = many(20, confidence=0.95, accuracy=0.8)
        report = calibrate(got, bin_count=10)
        payload = export.report_payload(report, got, choose_threshold(got))
        assert payload["summary"]["bin_count"] == 10


class TestChart:
    @requires_matplotlib
    def test_a_png_is_written(self, tmp_path):
        got = many(40, confidence=0.9, accuracy=0.7)
        path = export_path = tmp_path / "r.png"
        from ai_classifier.render.chart import write_chart

        write_chart(calibrate(got), path)
        assert export_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    @requires_matplotlib
    def test_an_unmeasurable_subgroup_draws_no_curve_and_says_so(self, tmp_path):
        """A flat line at zero would read as 'always wrong' -- a third misreading."""
        from ai_classifier.render.chart import write_chart

        path = write_chart(calibrate(abstentions(20)), tmp_path / "empty.png")
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
