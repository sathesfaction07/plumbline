"""Calibration: binning, ECE, the empty-denominator regression, thresholds, blind spots."""

from __future__ import annotations

import pytest

from ai_classifier.core.calibrate import (
    MIN_SAMPLES_PER_BIN,
    BinScheme,
    blind_spots,
    calibrate,
    calibrate_by_modality,
    choose_threshold,
    confusion,
    expected_calibration_error,
    reliability_bins,
)
from ai_classifier.core.feedback import apply_feedback, threshold_stability
from ai_classifier.models import Label, Modality, Verdict
from tests.conftest import abstentions, many


class TestBinning:
    def test_confidence_of_exactly_one_lands_in_the_last_bin(self):
        """int(1.0 * 10) is 10, one past the end. Providers do emit 1.0."""
        bins = reliability_bins(many(12, confidence=1.0, accuracy=1.0), bin_count=10)
        assert len(bins) == 1
        assert bins[0].count == 12

    def test_confidence_of_zero_lands_in_the_first_bin(self):
        got = many(12, confidence=0.0, accuracy=0.5)
        # A confidence of 0.0 on an ANSWER is odd but legal; only abstentions are pinned.
        bins = reliability_bins(got, bin_count=10)
        assert bins[0].lower == 0.0

    def test_equal_width_bins_can_be_empty_and_empty_ones_are_dropped(self):
        got = many(20, confidence=0.95, accuracy=0.8) + many(
            20, confidence=0.15, accuracy=0.2, prefix="Q"
        )
        bins = reliability_bins(got, bin_count=10)
        assert len(bins) == 2, "only two bands are populated"
        assert sum(item.count for item in bins) == 40

    def test_equal_count_bins_are_balanced(self):
        got = many(50, confidence=0.9, accuracy=0.8) + many(
            50, confidence=0.6, accuracy=0.6, prefix="Q"
        )
        bins = reliability_bins(got, bin_count=10, scheme=BinScheme.EQUAL_COUNT)
        counts = {item.count for item in bins}
        assert counts == {10}, counts

    def test_gap_is_signed_so_the_two_directions_are_distinguishable(self):
        over = reliability_bins(many(20, confidence=0.95, accuracy=0.6))[0]
        under = reliability_bins(many(20, confidence=0.60, accuracy=0.95))[0]
        assert over.gap > 0
        assert under.gap < 0

    def test_a_thin_bin_is_reported_but_flagged_unreliable(self):
        bins = reliability_bins(many(3, confidence=0.9, accuracy=1.0), bin_count=10)
        assert bins[0].count == 3
        assert not bins[0].is_reliable

    def test_a_bin_at_the_threshold_is_reliable(self):
        bins = reliability_bins(
            many(MIN_SAMPLES_PER_BIN, confidence=0.9, accuracy=0.9), bin_count=10
        )
        assert bins[0].is_reliable


class TestAbstentionsAreExcludedFromCalibration:
    """An abstention has no answer whose correctness could be compared to a confidence."""

    def test_abstentions_never_appear_in_a_bin(self):
        got = many(20, confidence=0.9, accuracy=0.8) + abstentions(30)
        report = calibrate(got)
        assert sum(item.count for item in report.bins) == 20
        assert report.answered == 20
        assert report.abstained == 30

    def test_adding_abstentions_does_not_move_the_ece(self):
        """Otherwise a provider could improve its calibration score by refusing to answer."""
        answered = many(40, confidence=0.95, accuracy=0.7)
        before = calibrate(answered).ece
        after = calibrate(answered + abstentions(200)).ece
        assert before == after

    def test_accuracy_is_over_answers_only(self):
        got = many(20, confidence=0.9, accuracy=1.0) + abstentions(80)
        report = calibrate(got)
        assert report.accuracy == 1.0, "abstentions must not count as errors"
        assert report.coverage == pytest.approx(0.2)
        assert report.abstention_rate == pytest.approx(0.8)


class TestEmptyDenominatorMustNotLookLikeAPerfectScore:
    """The regression.

    ECE is a distance, so 0.0 is the BEST attainable value -- and it is exactly what a naive
    implementation returns for a subgroup that was never answered. Ranking modalities by ECE
    put the two the provider is blind to at the top.
    """

    def test_ece_is_none_not_zero_when_nothing_was_answered(self):
        report = calibrate(abstentions(50))
        assert report.ece is None, "0.0 would be the best possible score"
        assert report.accuracy is None, "0.0 would read as 'got everything wrong'"
        assert not report.measurable

    def test_expected_calibration_error_of_no_bins_is_none(self):
        assert expected_calibration_error([]) is None

    def test_a_provider_that_abstains_on_everything_gets_no_score_at_all(self):
        report = calibrate(abstentions(280))
        assert report.ece is None
        assert report.abstained == 280
        assert report.coverage == 0.0

    def test_an_unmeasured_subgroup_cannot_be_sorted_against_a_measured_one(self):
        """The strongest guarantee available: the comparison raises rather than misleads."""
        got = many(30, confidence=0.95, accuracy=0.7, modality=Modality.TEXT_ONLY)
        got += abstentions(20, modality=Modality.IMAGE_ONLY)
        reports = calibrate_by_modality(got)
        assert reports[Modality.IMAGE_ONLY].ece is None
        assert reports[Modality.TEXT_ONLY].ece is not None
        with pytest.raises(TypeError):
            sorted(reports.values(), key=lambda item: item.ece)

    def test_the_two_headline_metrics_no_longer_contradict_each_other(self):
        """Before the fix, ECE said perfect and accuracy said total failure. Same cause."""
        report = calibrate(abstentions(43, modality=Modality.IMAGE_ONLY))
        assert report.ece is None and report.accuracy is None

    def test_measurable_is_the_flag_to_check_first(self):
        assert calibrate(many(12, confidence=0.9, accuracy=0.9)).measurable
        assert not calibrate(abstentions(12)).measurable
        assert not calibrate([]).measurable


class TestBlindSpots:
    def test_a_subgroup_with_no_answers_is_named(self):
        got = many(40, confidence=0.9, accuracy=0.8, modality=Modality.BOTH)
        got += abstentions(43, modality=Modality.IMAGE_ONLY)
        spots = blind_spots(calibrate_by_modality(got))
        assert [item.subgroup for item in spots] == ["image_only"]
        assert "NONE answered" in spots[0].describe()

    def test_a_thinly_answered_subgroup_is_also_named_with_a_different_message(self):
        got = many(40, confidence=0.9, accuracy=0.8, modality=Modality.BOTH)
        got += many(3, confidence=0.9, accuracy=1.0, modality=Modality.IMAGE_ONLY, prefix="I")
        got += abstentions(20, modality=Modality.IMAGE_ONLY)
        spots = blind_spots(calibrate_by_modality(got))
        assert [item.subgroup for item in spots] == ["image_only"]
        assert "only 3 answered" in spots[0].describe()

    def test_the_largest_blind_spot_comes_first(self):
        got = many(40, confidence=0.9, accuracy=0.8, modality=Modality.BOTH)
        got += abstentions(7, modality=Modality.NEITHER, prefix="N")
        got += abstentions(43, modality=Modality.IMAGE_ONLY, prefix="I")
        spots = blind_spots(calibrate_by_modality(got))
        assert [item.subgroup for item in spots] == ["image_only", "neither"]

    def test_no_blind_spots_when_everything_was_answered(self):
        got = many(40, confidence=0.9, accuracy=0.8, modality=Modality.BOTH)
        assert blind_spots(calibrate_by_modality(got)) == []


class TestThreshold:
    def test_an_unreachable_target_returns_none_rather_than_a_usable_looking_number(self):
        """A sentinel of 1.01 was the first attempt and violated the field's own le=1.0."""
        got = many(40, confidence=0.95, accuracy=0.60)
        threshold = choose_threshold(got, target_accuracy=0.95)
        assert threshold.value is None
        assert not threshold.reachable
        assert threshold.auto_accepted == 0
        assert "nothing should be auto-accepted" in threshold.basis

    def test_a_reachable_target_picks_the_lowest_qualifying_cut_off(self):
        got = many(30, confidence=0.99, accuracy=1.0)
        got += many(30, confidence=0.80, accuracy=1.0, prefix="Q")
        got += many(30, confidence=0.50, accuracy=0.2, prefix="R")
        threshold = choose_threshold(got, target_accuracy=0.95)
        assert threshold.value == 0.8, "0.8 and above is 100% accurate"
        assert threshold.auto_accepted == 60

    def test_the_threshold_is_chosen_on_observed_accuracy_not_stated_confidence(self):
        """A provider claiming 0.99 while being right 50% of the time must not qualify."""
        got = many(40, confidence=0.99, accuracy=0.50)
        assert choose_threshold(got, target_accuracy=0.95).value is None

    def test_a_population_too_small_to_judge_is_skipped(self):
        got = many(4, confidence=0.99, accuracy=1.0)
        assert choose_threshold(got, target_accuracy=0.95).value is None

    def test_nothing_answered_is_reported_as_such(self):
        threshold = choose_threshold(abstentions(20), target_accuracy=0.95)
        assert threshold.value is None
        assert "nothing was answered" in threshold.basis
        assert threshold.routed_to_human == 20


class TestConfusionIncludesAbstentions:
    def test_abstentions_appear_so_their_shape_is_visible(self):
        got = many(10, confidence=0.9, accuracy=1.0) + abstentions(5)
        counts = confusion(got)
        assert counts[(Verdict.TRUE_POSITIVE, Verdict.TRUE_POSITIVE)] == 10
        assert counts[(Verdict.NOT_ENOUGH_INFO, Verdict.TRUE_POSITIVE)] == 5


class TestBinCountIsCarriedNotRederived:
    def test_the_report_remembers_what_was_asked_for(self):
        """len(bins) counts POPULATED bins, so a caller re-deriving it rebins differently."""
        got = many(20, confidence=0.95, accuracy=0.8)
        report = calibrate(got, bin_count=10)
        assert report.bin_count == 10
        assert len(report.bins) == 1


class TestFeedback:
    def test_a_new_verdict_is_added(self):
        incoming = [Label(product_id="P-1", verdict=Verdict.TRUE_POSITIVE)]
        merged, report = apply_feedback([], incoming)
        assert report.added == ["P-1"]
        assert len(merged) == 1

    def test_an_agreeing_verdict_is_unchanged(self):
        existing = [Label(product_id="P-1", verdict=Verdict.TRUE_POSITIVE)]
        _, report = apply_feedback(existing, existing)
        assert report.unchanged == ["P-1"]
        assert report.overturned == []

    def test_a_contradicting_verdict_is_reported_and_the_new_one_wins(self):
        existing = [Label(product_id="P-1", verdict=Verdict.TRUE_POSITIVE)]
        incoming = [Label(product_id="P-1", verdict=Verdict.FALSE_POSITIVE)]
        merged, report = apply_feedback(existing, incoming)
        assert report.overturned == ["P-1"]
        assert merged[0].verdict is Verdict.FALSE_POSITIVE

    def test_stability_reports_when_the_recommendation_stopped_moving(self):
        got = many(80, confidence=0.99, accuracy=1.0)
        report = threshold_stability(got, target_accuracy=0.95, steps=8)
        assert report.stable
        assert report.settled_after is not None

    def test_an_unstable_recommendation_says_so(self):
        """Early labels all correct, later ones not: the answer changes at the end."""
        got = many(40, confidence=0.99, accuracy=1.0)
        got += many(40, confidence=0.99, accuracy=0.0, prefix="Q")
        report = threshold_stability(got, target_accuracy=0.95, steps=8)
        assert report.final.value is None
        assert report.points[0].threshold is not None, "it did qualify early on"

    def test_no_outcomes_produces_no_points(self):
        report = threshold_stability([], target_accuracy=0.95)
        assert report.points == []
        assert not report.stable


class TestPropertiesOverTheShippedCatalogue:
    """Run against data/, so editing it and breaking an invariant fails the build."""

    def test_every_outcome_is_paired_with_its_own_label(self, outcomes):
        for item in outcomes:
            assert item.classification.product_id == item.label.product_id

    def test_bins_hold_exactly_the_answered_outcomes(self, outcomes):
        report = calibrate(outcomes)
        assert sum(item.count for item in report.bins) == report.answered

    def test_the_modality_breakdown_partitions_the_population(self, outcomes):
        reports = calibrate_by_modality(outcomes)
        assert sum(item.total for item in reports.values()) == len(outcomes)
        assert sum(item.answered for item in reports.values()) == sum(
            1 for item in outcomes if item.answered
        )

    def test_the_offline_provider_is_blind_to_image_only_listings(self, outcomes):
        """Not a defect: it cannot see pictures. The point is that it is REPORTED."""
        reports = calibrate_by_modality(outcomes)
        image_only = reports[Modality.IMAGE_ONLY]
        assert image_only.total > 0, "the catalogue must contain image-only listings"
        assert image_only.answered == 0
        assert image_only.ece is None
        assert any(item.subgroup == "image_only" for item in blind_spots(reports))

    def test_both_binning_schemes_agree_on_the_direction(self, outcomes):
        wide = calibrate(outcomes, scheme=BinScheme.EQUAL_WIDTH)
        even = calibrate(outcomes, scheme=BinScheme.EQUAL_COUNT)
        assert wide.direction == even.direction
        assert abs(wide.ece - even.ece) < 0.05, "schemes may differ, but not qualitatively"

    def test_calibration_is_order_independent(self, outcomes):
        import random

        reference = calibrate(outcomes)
        for seed in range(6):
            shuffled = list(outcomes)
            random.Random(seed).shuffle(shuffled)
            got = calibrate(shuffled)
            assert got.ece == reference.ece
            assert got.answered == reference.answered
            assert [item.count for item in got.bins] == [
                item.count for item in reference.bins
            ]

    def test_the_shipped_catalogue_exercises_every_modality(self, outcomes):
        seen = {item.classification.modality for item in outcomes}
        assert seen == set(Modality), f"missing {set(Modality) - seen}"

    def test_the_shipped_catalogue_exercises_every_verdict(self, outcomes):
        predicted = {item.classification.verdict for item in outcomes}
        assert predicted == set(Verdict), f"missing {set(Verdict) - predicted}"
