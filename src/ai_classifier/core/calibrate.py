"""Measuring whether a stated confidence means anything.

A provider that reports 0.95 on every classification is not thereby 95% accurate. The
number is a claim, and this module is the only place that checks it against ground truth.

The distinction that organises everything here: ACCURACY asks "how often is it right",
CALIBRATION asks "when it says 0.9, is it right 90% of the time". They are independent.
A classifier can be 97% accurate and badly calibrated (it says 0.99 on everything, so its
confidence carries no information about which 3% to review), or 70% accurate and perfectly
calibrated (it says 0.7 and is right 70% of the time, which is genuinely useful -- you
know exactly what to trust).

Only the second property lets you automate anything. Routing on an uncalibrated confidence
means auto-accepting a population whose real error rate you do not know.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, Field

from ai_classifier.models import Modality, Outcome, Verdict


class BinScheme(StrEnum):
    """How to divide the confidence range into bins.

    Both are standard and they disagree, sometimes sharply, on the same data. Offering
    both -- and reporting which was used -- is the honest option, because a single ECE
    number carries an unstated binning choice that materially affects it.
    """

    EQUAL_WIDTH = "equal_width"
    EQUAL_COUNT = "equal_count"


class Bin(BaseModel):
    """One confidence band, with what actually happened inside it."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    observed_accuracy: float

    @property
    def gap(self) -> float:
        """Signed: positive means overconfident, negative means underconfident.

        Signed rather than absolute because the two have opposite remedies. Overconfident
        means raise the auto-accept threshold; underconfident means work is being sent to
        humans that did not need to go there.
        """
        return self.mean_confidence - self.observed_accuracy

    @property
    def is_reliable(self) -> bool:
        """Whether this bin holds enough samples to say anything.

        A bin with one sample has an observed accuracy of exactly 0.0 or 1.0, which looks
        like a catastrophic or perfect result and is neither. Ten is a low bar chosen to
        be honest rather than flattering: below it, the bin is reported but excluded from
        claims about the provider.
        """
        return self.count >= MIN_SAMPLES_PER_BIN


MIN_SAMPLES_PER_BIN = 10


class CalibrationReport(BaseModel):
    """What the measurement found.

    Carries the sample count and the binning scheme alongside the numbers, because an ECE
    quoted without them is not reproducible.

    **`ece` and `accuracy` are None when nothing was answered, and that is deliberate.**
    ECE is a distance, so its best possible value is 0.0 -- and 0.0 is exactly what a
    naive implementation returns for a subgroup the classifier never answered, because
    there are no bins to disagree with reality. Measured on real data, ranking modalities
    by ECE put the two the provider is completely blind to at the TOP:

        ECE=0.0000  image_only  (answered   0 of  43)   <- best calibrated?
        ECE=0.0000  neither     (answered   0 of   7)   <- best calibrated?
        ECE=0.0802  text_only   (answered  80 of  82)
        ECE=0.1574  both        (answered 143 of 148)

    A provider abstaining on all 280 items scored a perfect headline ECE. Worse, on the
    same subgroup `accuracy` returned 0.0, which reads as total failure -- so the two
    headline metrics pointed in opposite directions and both were artefacts of the same
    empty denominator.

    None makes the undefined case unrepresentable as a number, so it cannot be sorted,
    averaged, or plotted next to a real one by accident. `measurable` says which you have.
    """

    scheme: BinScheme
    # The number of bins REQUESTED, which is not len(bins): only populated bins are kept,
    # and with equal-width binning over saturating confidences most of the range is empty.
    # Carried explicitly because a caller re-deriving it from len(bins) silently rebins the
    # breakdown at a different granularity than the headline it sits under.
    bin_count: int
    bins: list[Bin]
    answered: int
    abstained: int
    correct: int
    ece: float | None
    subgroup: str = "all"

    @property
    def total(self) -> int:
        return self.answered + self.abstained

    @property
    def measurable(self) -> bool:
        """Whether anything was answered, so whether any metric here means anything.

        Read this before reading `ece` or `accuracy`. Both are None when it is False, and
        that is the whole point -- see the class docstring.
        """
        return self.answered > 0

    @property
    def accuracy(self) -> float | None:
        """Accuracy over ANSWERS ONLY, or None if nothing was answered.

        Abstentions are excluded rather than counted as errors. A classifier that declines
        a case it cannot see has not made a mistake -- it has correctly identified that it
        needs help. Counting that as wrong makes an honest abstainer look worse than a
        confident guesser, which is precisely backwards for a review pipeline.

        The cost is that this number alone can be gamed: abstain on everything hard and
        accuracy goes up. That is why `abstention_rate` sits next to it and why `coverage`
        exists -- the pair cannot be gamed, only traded off.

        None rather than 0.0 for an empty answered set. 0.0 reads as "got everything
        wrong", which is a measurement, when the truth is "there is nothing to measure".
        """
        if self.answered == 0:
            return None
        return self.correct / self.answered

    @property
    def abstention_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.abstained / self.total

    @property
    def coverage(self) -> float:
        """Fraction of the population the classifier was willing to answer at all."""
        if self.total == 0:
            return 0.0
        return self.answered / self.total

    @property
    def reliable_bins(self) -> list[Bin]:
        return [item for item in self.bins if item.is_reliable]

    @property
    def direction(self) -> str:
        """Which way the miscalibration runs, over reliable bins only."""
        usable = self.reliable_bins
        if not usable:
            return "unknown"
        weighted = sum(item.gap * item.count for item in usable)
        total = sum(item.count for item in usable)
        mean_gap = weighted / total
        if abs(mean_gap) < 0.02:
            return "calibrated"
        return "overconfident" if mean_gap > 0 else "underconfident"


def _bin_index_equal_width(confidence: float, bin_count: int) -> int:
    """Which equal-width bin a confidence falls in.

    The `min` is not defensive clutter. `int(1.0 * 10)` is 10, which is one past the last
    bin, and a confidence of exactly 1.0 is not hypothetical -- providers emit it and
    round-tripping through JSON produces it. Without the clamp this raises IndexError on
    the most confident item in the set, which is the one a reviewer most wants to see.
    """
    return min(int(confidence * bin_count), bin_count - 1)


def reliability_bins(
    outcomes: list[Outcome],
    *,
    bin_count: int = 10,
    scheme: BinScheme = BinScheme.EQUAL_WIDTH,
) -> list[Bin]:
    """Group answered outcomes by stated confidence and measure accuracy in each.

    ABSTENTIONS ARE EXCLUDED. There is no answer whose correctness could be compared
    against a confidence, so an abstention has nothing to contribute to a reliability
    curve. Including them would put items in the low-confidence bins that were never
    answers, and low accuracy in a low-confidence bin reads as *good* calibration -- so a
    provider that abstains heavily would score as better calibrated than one that
    commits. The abstention rate is reported separately instead.
    """
    answered = [item for item in outcomes if item.answered]
    if not answered:
        return []

    if scheme is BinScheme.EQUAL_WIDTH:
        grouped: list[list[Outcome]] = [[] for _ in range(bin_count)]
        for item in answered:
            grouped[_bin_index_equal_width(item.confidence, bin_count)].append(item)
        edges = [(index / bin_count, (index + 1) / bin_count) for index in range(bin_count)]
    else:
        ordered = sorted(answered, key=lambda item: item.confidence)
        # Slice as evenly as the count allows; the first few bins absorb the remainder so
        # that every bin is non-empty when there are at least `bin_count` samples.
        size, extra = divmod(len(ordered), bin_count)
        grouped = []
        cursor = 0
        for index in range(bin_count):
            take = size + (1 if index < extra else 0)
            grouped.append(ordered[cursor : cursor + take])
            cursor += take
        edges = [
            (group[0].confidence, group[-1].confidence) if group else (0.0, 0.0)
            for group in grouped
        ]

    result = []
    for (lower, upper), group in zip(edges, grouped, strict=True):
        if not group:
            continue
        result.append(
            Bin(
                lower=lower,
                upper=upper,
                count=len(group),
                mean_confidence=sum(item.confidence for item in group) / len(group),
                observed_accuracy=sum(1 for item in group if item.correct) / len(group),
            )
        )
    return result


def expected_calibration_error(bins: list[Bin]) -> float | None:
    """Mean absolute gap between claimed and observed, weighted by bin population.

    Weighted, not a plain mean over bins. An unweighted mean lets a bin holding three
    samples count as much as one holding three hundred, which with equal-width binning
    means the headline number is dominated by the sparsest, noisiest bands -- exactly
    where confidences are rare. Weighting is the standard definition and also the one
    that does not move when you add an empty bin.

    Returns None, not 0.0, for an empty set of bins. Zero is the BEST attainable score
    here, so returning it for "nothing was measured" does not merely lose information --
    it awards a perfect result for having answered nothing at all.
    """
    total = sum(item.count for item in bins)
    if total == 0:
        return None
    return sum(abs(item.gap) * item.count for item in bins) / total


def calibrate(
    outcomes: list[Outcome],
    *,
    bin_count: int = 10,
    scheme: BinScheme = BinScheme.EQUAL_WIDTH,
    subgroup: str = "all",
) -> CalibrationReport:
    """Measure a set of outcomes."""
    bins = reliability_bins(outcomes, bin_count=bin_count, scheme=scheme)
    answered = [item for item in outcomes if item.answered]
    return CalibrationReport(
        scheme=scheme,
        bin_count=bin_count,
        bins=bins,
        answered=len(answered),
        abstained=len(outcomes) - len(answered),
        correct=sum(1 for item in answered if item.correct),
        ece=expected_calibration_error(bins),
        subgroup=subgroup,
    )


def calibrate_by_modality(
    outcomes: list[Outcome],
    *,
    bin_count: int = 10,
    scheme: BinScheme = BinScheme.EQUAL_WIDTH,
) -> dict[Modality, CalibrationReport]:
    """Measure each modality separately.

    This exists because the aggregate hides the case this tool was built for. Image-only
    listings are a minority of any catalogue, so their contribution to a
    population-weighted ECE is proportionally small -- a provider can be well calibrated
    overall and badly miscalibrated on exactly the subgroup that motivated the work,
    with the headline number showing nothing.
    """
    grouped: dict[Modality, list[Outcome]] = {}
    for item in outcomes:
        grouped.setdefault(item.classification.modality, []).append(item)
    return {
        modality: calibrate(
            group, bin_count=bin_count, scheme=scheme, subgroup=str(modality.value)
        )
        for modality, group in sorted(grouped.items())
    }


class BlindSpot(BaseModel):
    """A subgroup the provider declined to answer often enough to be unmeasurable."""

    subgroup: str
    total: int
    answered: int

    @property
    def coverage(self) -> float:
        return self.answered / self.total if self.total else 0.0

    def describe(self) -> str:
        if self.answered == 0:
            return (
                f"{self.subgroup}: {self.total} item(s), NONE answered. No calibration "
                f"figure exists for this subgroup -- it was not measured, and a reported "
                f"0.0 would mean the opposite."
            )
        return (
            f"{self.subgroup}: {self.total} item(s), only {self.answered} answered "
            f"({self.coverage:.0%}); too few for a trustworthy figure."
        )


def blind_spots(
    reports: dict[Modality, CalibrationReport],
    *,
    min_answered: int = MIN_SAMPLES_PER_BIN,
) -> list[BlindSpot]:
    """Subgroups whose numbers should not be read as measurements.

    This is the actionable half of the fix that made `ece` optional. Turning the undefined
    case into None stops it being mistaken for a good score, but None on its own is easy to
    skim past in a table -- and the subgroup a reviewer most needs to hear about is exactly
    the one with nothing in it. Naming them explicitly turns silence into a finding.

    Sorted by how much is unmeasured, largest first, because a blind spot covering 43 items
    matters more than one covering 7.
    """
    found = [
        BlindSpot(subgroup=report.subgroup, total=report.total, answered=report.answered)
        for report in reports.values()
        if report.answered < min_answered
    ]
    return sorted(found, key=lambda item: (-(item.total - item.answered), item.subgroup))


def confusion(outcomes: list[Outcome]) -> dict[tuple[Verdict, Verdict], int]:
    """Counts of (predicted, actual), abstentions included.

    Abstentions belong here even though they are excluded from calibration: the shape of
    what a classifier declines is diagnostic. Abstaining mostly on items whose true label
    is FALSE_POSITIVE is a different problem from abstaining at random.
    """
    return Counter(
        (item.classification.verdict, item.label.verdict) for item in outcomes
    )


class Threshold(BaseModel):
    """A confidence cut-off for auto-accepting a classification.

    `value` is None when NO cut-off reaches the target accuracy, meaning nothing should be
    automated. An out-of-range sentinel was the first attempt -- 1.01, on the reasoning
    that no real confidence could ever exceed it -- and it contradicted this model's own
    `le=1.0` constraint, so the one branch that most needed to be reportable raised a
    validation error instead. None says the same thing without lying about the type, and a
    caller cannot compare against it by accident.
    """

    value: float | None = Field(default=None, ge=0.0, le=1.0)
    target_accuracy: float
    achieved_accuracy: float
    auto_accepted: int
    routed_to_human: int
    basis: str

    @property
    def reachable(self) -> bool:
        return self.value is not None

    @property
    def automation_rate(self) -> float:
        total = self.auto_accepted + self.routed_to_human
        return self.auto_accepted / total if total else 0.0


def choose_threshold(
    outcomes: list[Outcome],
    *,
    target_accuracy: float = 0.95,
) -> Threshold:
    """Lowest confidence at which observed accuracy meets the target.

    Chosen on OBSERVED accuracy at or above each candidate, not on the stated confidence.
    Using the stated number would assume the calibration this function exists to work
    around: if a provider claims 0.95 while being right 80% of the time, a threshold of
    0.95 auto-accepts a population with a 20% error rate.

    Scanning from the lowest qualifying candidate maximises how much gets automated. The
    curve is not guaranteed monotonic -- with real data accuracy at 0.8 can exceed
    accuracy at 0.9 -- so "lowest that qualifies" and "highest that qualifies" are
    different answers, and which one is chosen is recorded in `basis`.

    Returns `value=None` when no cut-off reaches the target, which means nothing should
    be automated. That is a legitimate answer and callers must handle it rather than
    receiving a plausible-looking number.
    """
    answered = [item for item in outcomes if item.answered]
    if not answered:
        return Threshold(
            value=None,
            target_accuracy=target_accuracy,
            achieved_accuracy=0.0,
            auto_accepted=0,
            routed_to_human=len(outcomes),
            basis="nothing was answered, so no threshold can be measured",
        )

    candidates = sorted({round(item.confidence, 2) for item in answered})
    for candidate in candidates:
        at_or_above = [item for item in answered if item.confidence >= candidate]
        if len(at_or_above) < MIN_SAMPLES_PER_BIN:
            continue
        accuracy = sum(1 for item in at_or_above if item.correct) / len(at_or_above)
        if accuracy >= target_accuracy:
            return Threshold(
                value=candidate,
                target_accuracy=target_accuracy,
                achieved_accuracy=accuracy,
                auto_accepted=len(at_or_above),
                routed_to_human=len(outcomes) - len(at_or_above),
                basis=(
                    f"lowest confidence whose population of {len(at_or_above)} reaches "
                    f"{accuracy:.1%}"
                ),
            )

    best = sum(1 for item in answered if item.correct) / len(answered)
    return Threshold(
        value=None,
        target_accuracy=target_accuracy,
        achieved_accuracy=best,
        auto_accepted=0,
        routed_to_human=len(outcomes),
        basis=(
            f"no cut-off reaches {target_accuracy:.0%}; the whole answered population "
            f"peaks at {best:.1%}, so nothing should be auto-accepted"
        ),
    )
