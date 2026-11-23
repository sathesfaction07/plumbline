"""Text rendering: a reliability diagram you can read in a terminal and assert in CI.

A reliability diagram is normally a plot, and a plot can only be checked by a human looking
at it. This is the same information as characters, so a test can pin it and a CI step can
grep it. The matplotlib version in `chart.py` is nicer to look at and exists alongside
rather than instead.

Everywhere a number could be absent it renders as `n/a` with the reason stated nearby.
That is the whole lesson of this codebase: `0.0000` in an ECE column is the best possible
score, so printing it for "never answered" is not a display nicety, it is a lie.
"""

from __future__ import annotations

from ai_classifier.core.calibrate import (
    BlindSpot,
    CalibrationReport,
    Threshold,
    blind_spots,
    calibrate_by_modality,
    confusion,
)
from ai_classifier.models import Modality, Outcome

NOT_MEASURED = "n/a"


def _pct(value: float | None) -> str:
    return NOT_MEASURED if value is None else f"{value:.1%}"


def _ece(value: float | None) -> str:
    return NOT_MEASURED if value is None else f"{value:.4f}"


def reliability_diagram(report: CalibrationReport, *, width: int = 40) -> list[str]:
    """The observed-accuracy curve against the claimed-confidence diagonal.

    Each row is one bin. `|` marks where claimed confidence sits and `#` where observed
    accuracy sits, so the distance between them IS the calibration gap -- the eye reads the
    error directly rather than comparing two numbers.
    """
    if not report.bins:
        return [
            "  no reliability curve: nothing was answered, so there is no claimed",
            "  confidence to compare against an outcome",
        ]

    lines = ["  bin        n     claimed  observed   gap  diagram"]
    for item in report.bins:
        claimed_at = min(int(item.mean_confidence * width), width - 1)
        observed_at = min(int(item.observed_accuracy * width), width - 1)
        row = [" "] * width
        row[claimed_at] = "|"
        # Observed wins the cell when they coincide: a perfectly calibrated bin should read
        # as a solid mark, not as a bar that happens to hide the outcome.
        row[observed_at] = "#"
        note = "" if item.is_reliable else "  (thin)"
        lines.append(
            f"  [{item.lower:.1f},{item.upper:.1f}) {item.count:>5}   "
            f"{item.mean_confidence:.3f}     {item.observed_accuracy:.3f}  "
            f"{item.gap:+.3f}  {''.join(row)}{note}"
        )
    lines.append(f"  {'':>39}| claimed   # observed")
    return lines


def summary_block(report: CalibrationReport) -> list[str]:
    """Headline numbers, with the unmeasurable ones named as such."""
    lines = [
        f"  items            {report.total}",
        f"  answered         {report.answered}  ({report.coverage:.1%} coverage)",
        f"  abstained        {report.abstained}  ({report.abstention_rate:.1%})",
        f"  accuracy         {_pct(report.accuracy)}   (over answers only)",
        f"  ECE              {_ece(report.ece)}   ({report.scheme.value} bins)",
        f"  direction        {report.direction}",
    ]
    if not report.measurable:
        lines.append(
            "  ^ nothing was answered, so accuracy and ECE do not exist for this set. "
            "A 0.0 here would read as a perfect result."
        )
    return lines


def modality_table(reports: dict[Modality, CalibrationReport]) -> list[str]:
    """Per-modality breakdown.

    The reason this table exists: the aggregate cannot show that a provider is blind to a
    whole subgroup, because a subgroup it never answered contributes nothing to the
    aggregate -- not a bad number, no number at all.
    """
    lines = ["  modality      items  answered  coverage  accuracy       ECE  direction"]
    for modality, report in reports.items():
        lines.append(
            f"  {modality.value:<12} {report.total:>6} {report.answered:>9} "
            f"{report.coverage:>9.1%} {_pct(report.accuracy):>9} "
            f"{_ece(report.ece):>9}  {report.direction}"
        )
    return lines


def blind_spot_block(spots: list[BlindSpot]) -> list[str]:
    if not spots:
        return ["  none: every modality had enough answers to measure"]
    return [f"  - {spot.describe()}" for spot in spots]


def threshold_block(threshold: Threshold) -> list[str]:
    """The routing recommendation.

    When no cut-off reaches the target the advice is "automate nothing", and that has to be
    stated as advice rather than shown as a threshold of 1.0 -- which a reader would take
    as "auto-accept anything at 100% confidence", a rule that would fire.
    """
    if not threshold.reachable:
        return [
            f"  target {threshold.target_accuracy:.0%}: NOT REACHABLE",
            f"  {threshold.basis}",
            f"  recommendation: route all {threshold.routed_to_human} to human review",
        ]
    return [
        f"  target {threshold.target_accuracy:.0%}: auto-accept at confidence "
        f">= {threshold.value:.2f}",
        f"  achieves {threshold.achieved_accuracy:.1%} over {threshold.auto_accepted} item(s)",
        f"  automates {threshold.automation_rate:.1%}, routes "
        f"{threshold.routed_to_human} to human review",
        f"  basis: {threshold.basis}",
    ]


def confusion_block(outcomes: list[Outcome]) -> list[str]:
    counts = confusion(outcomes)
    if not counts:
        return ["  nothing to show"]
    lines = ["  predicted          actual              n"]
    for (predicted, actual), count in sorted(counts.items()):
        lines.append(f"  {predicted.value:<18} {actual.value:<18} {count:>4}")
    return lines


def render(
    report: CalibrationReport,
    outcomes: list[Outcome],
    threshold: Threshold,
    *,
    show_confusion: bool = True,
) -> str:
    """The full text report."""
    # report.bin_count, NOT len(report.bins): the latter counts only populated bins, so the
    # breakdown would be binned more coarsely than the headline it appears beneath.
    by_modality = calibrate_by_modality(
        outcomes, bin_count=report.bin_count, scheme=report.scheme
    )
    sections: list[tuple[str, list[str]]] = [
        ("summary", summary_block(report)),
        ("reliability", reliability_diagram(report)),
        ("by modality", modality_table(by_modality)),
        ("blind spots", blind_spot_block(blind_spots(by_modality))),
        ("routing", threshold_block(threshold)),
    ]
    if show_confusion:
        sections.append(("confusion", confusion_block(outcomes)))

    lines: list[str] = []
    for title, body in sections:
        lines.append(title)
        lines.extend(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
