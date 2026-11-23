"""Machine-readable output: JSON Lines for piping, CSV for spreadsheets, JSON for a report.

One rule governs every writer here: a value that does not exist is written as null or an
empty cell, NEVER as 0. In an ECE column 0.0000 is the best attainable score, so writing it
for "never answered" would export the exact inversion this codebase was built to remove --
and a spreadsheet will happily sort on it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ai_classifier.core.calibrate import (
    CalibrationReport,
    Threshold,
    blind_spots,
    calibrate_by_modality,
)
from ai_classifier.models import Outcome

CLASSIFICATION_FIELDS = [
    "product_id", "verdict", "confidence", "modality", "provider", "rationale",
]
OUTCOME_FIELDS = CLASSIFICATION_FIELDS + ["label", "answered", "correct"]
BIN_FIELDS = [
    "subgroup", "lower", "upper", "count", "mean_confidence", "observed_accuracy",
    "gap", "is_reliable",
]


def outcome_rows(outcomes: list[Outcome]) -> list[dict]:
    """Flatten outcomes for export.

    `correct` is written as the string "abstained" for an abstention rather than as an empty
    cell. Empty reads as missing data; the classifier declining is a result, and the two
    must not look the same to whoever opens the file.
    """
    rows = []
    for item in outcomes:
        rows.append({
            "product_id": item.product_id,
            "verdict": item.classification.verdict.value,
            "confidence": item.confidence,
            "modality": item.classification.modality.value,
            "provider": item.classification.provider,
            "rationale": item.classification.rationale,
            "label": item.label.verdict.value,
            "answered": item.answered,
            "correct": "abstained" if item.correct is None else item.correct,
        })
    return rows


def bin_rows(reports: dict) -> list[dict]:
    rows = []
    for report in reports.values():
        for item in report.bins:
            rows.append({
                "subgroup": report.subgroup,
                "lower": round(item.lower, 4),
                "upper": round(item.upper, 4),
                "count": item.count,
                "mean_confidence": round(item.mean_confidence, 4),
                "observed_accuracy": round(item.observed_accuracy, 4),
                "gap": round(item.gap, 4),
                "is_reliable": item.is_reliable,
            })
    return rows


def report_payload(
    report: CalibrationReport, outcomes: list[Outcome], threshold: Threshold
) -> dict:
    """The JSON report.

    `ece` and `accuracy` serialise as null when unmeasurable. A consumer must then decide
    what to do, which is the correct outcome -- `json.load(...)["ece"] < 0.05` evaluating
    True on a subgroup nobody answered is how the original defect would have escaped into
    a dashboard.
    """
    by_modality = calibrate_by_modality(
        outcomes, bin_count=report.bin_count, scheme=report.scheme
    )
    return {
        "summary": {
            "items": report.total,
            "answered": report.answered,
            "abstained": report.abstained,
            "coverage": round(report.coverage, 4),
            "accuracy": None if report.accuracy is None else round(report.accuracy, 4),
            "ece": None if report.ece is None else round(report.ece, 4),
            "measurable": report.measurable,
            "direction": report.direction,
            "scheme": report.scheme.value,
            "bin_count": report.bin_count,
        },
        "by_modality": {
            modality.value: {
                "items": item.total,
                "answered": item.answered,
                "coverage": round(item.coverage, 4),
                "accuracy": None if item.accuracy is None else round(item.accuracy, 4),
                "ece": None if item.ece is None else round(item.ece, 4),
                "measurable": item.measurable,
                "direction": item.direction,
            }
            for modality, item in by_modality.items()
        },
        "blind_spots": [
            {"subgroup": s.subgroup, "items": s.total, "answered": s.answered,
             "note": s.describe()}
            for s in blind_spots(by_modality)
        ],
        "threshold": {
            "value": threshold.value,
            "reachable": threshold.reachable,
            "target_accuracy": threshold.target_accuracy,
            "achieved_accuracy": round(threshold.achieved_accuracy, 4),
            "auto_accepted": threshold.auto_accepted,
            "routed_to_human": threshold.routed_to_human,
            "automation_rate": round(threshold.automation_rate, 4),
            "basis": threshold.basis,
        },
    }


def write_jsonl(rows: list[dict], path: Path) -> Path:
    """Write one JSON object per line. Materialises the rows first, so not a stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> Path:
    """Write a CSV, header included even when there are no rows.

    A header-only file says "measured, found nothing". A missing file cannot be told apart
    from a run that died before reaching this point.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_all(
    report: CalibrationReport,
    outcomes: list[Outcome],
    threshold: Threshold,
    directory: Path,
) -> list[Path]:
    """Write every artefact. All files always appear, empty if need be."""
    directory.mkdir(parents=True, exist_ok=True)
    by_modality = calibrate_by_modality(
        outcomes, bin_count=report.bin_count, scheme=report.scheme
    )
    written = [
        write_csv(outcome_rows(outcomes), directory / "outcomes.csv", OUTCOME_FIELDS),
        write_csv(bin_rows(by_modality), directory / "bins.csv", BIN_FIELDS),
        write_jsonl(outcome_rows(outcomes), directory / "outcomes.jsonl"),
    ]
    payload = directory / "report.json"
    payload.write_text(
        json.dumps(report_payload(report, outcomes, threshold), indent=2) + "\n",
        encoding="utf-8",
    )
    written.append(payload)
    return written
