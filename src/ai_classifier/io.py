"""Loading products and labels, and pairing classifications with ground truth.

The pairing is the part worth reading. A calibration measurement is only meaningful over
items that have BOTH a classification and a label, and the set of such items is smaller
than either input -- usually much smaller, because labelling is expensive. Every function
here reports what it could not match instead of quietly returning the intersection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel

from ai_classifier.models import Classification, Label, Outcome, Product, Verdict

PRODUCT_ID_KEYS = ("product_id", "id", "item_id", "sku")
IMAGE_KEYS = ("images", "image_urls", "image")


def _first_present(row: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def product_from_row(row: dict) -> Product:
    """Build a Product from a feed row.

    Everything that is not an id or an image reference becomes an attribute. That is
    deliberately permissive: feeds disagree about which fields exist, and a whitelist would
    silently discard the one column that carried the deciding evidence.
    """
    product_id = _first_present(row, PRODUCT_ID_KEYS)
    if not product_id:
        raise ValueError(f"row has no product id (looked for {', '.join(PRODUCT_ID_KEYS)})")

    raw_images = _first_present(row, IMAGE_KEYS) or ""
    images = [part.strip() for part in raw_images.replace("|", ",").split(",") if part.strip()]

    skip = set(PRODUCT_ID_KEYS) | set(IMAGE_KEYS)
    attributes = {
        key: str(value)
        for key, value in row.items()
        if key not in skip and value is not None and str(value).strip()
    }
    return Product(product_id=product_id, attributes=attributes, images=images)


class LoadReport(BaseModel):
    """What loading a file produced, including what it refused."""

    products: list[Product] = []
    labels: list[Label] = []
    problems: list[str] = []
    undecided_labels: list[str] = []


def load_products(path: Path) -> LoadReport:
    """Load products from CSV or JSON Lines.

    utf-8-sig because a spreadsheet export carries a byte-order mark, and without it the
    first column name arrives as "\\ufeffproduct_id" -- so the id lookup fails on every row
    of an otherwise perfectly good file.
    """
    report = LoadReport()
    rows = _read_rows(path)
    for number, row in rows:
        try:
            report.products.append(product_from_row(row))
        except ValueError as error:
            report.problems.append(f"{path.name}:{number}: {error}")
    return report


def load_labels(path: Path) -> LoadReport:
    """Load ground-truth labels.

    A label of `not_enough_info` is collected into `undecided_labels` rather than being
    dropped or raising. It is real data -- it marks a product a reviewer could not settle --
    but it is not a target to measure against, so it must leave the scoring set while
    remaining visible. Silently dropping it would shrink the denominator without saying so.
    """
    report = LoadReport()
    for number, row in _read_rows(path):
        product_id = _first_present(row, PRODUCT_ID_KEYS)
        raw = (row.get("verdict") or row.get("label") or "").strip()
        if not product_id or not raw:
            report.problems.append(f"{path.name}:{number}: needs a product id and a verdict")
            continue
        try:
            verdict = Verdict(raw)
        except ValueError:
            report.problems.append(f"{path.name}:{number}: {raw!r} is not a verdict")
            continue
        if verdict is Verdict.NOT_ENOUGH_INFO:
            report.undecided_labels.append(product_id)
            continue
        report.labels.append(
            Label(
                product_id=product_id,
                verdict=verdict,
                source=str(row.get("source") or "human"),
            )
        )
    return report


def _read_rows(path: Path) -> list[tuple[int, dict]]:
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                rows.append((number, json.loads(line)))
        return rows
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{path.name}: expected a JSON array of objects")
        return list(enumerate(payload, start=1))
    reader = csv.DictReader(text.splitlines())
    return list(enumerate(reader, start=2))


class PairReport(BaseModel):
    """Classifications joined to labels, with both kinds of miss reported.

    `unlabelled` and `unclassified` are not errors, and both are normal. But they are the
    difference between "we measured the provider" and "we measured the provider on the
    subset somebody happened to label", which are different claims. A calibration report
    quoting a sample size without them is quoting a number nobody can reproduce.
    """

    outcomes: list[Outcome] = []
    unlabelled: list[str] = []
    unclassified: list[str] = []

    @property
    def coverage(self) -> float:
        total = len(self.outcomes) + len(self.unlabelled)
        return len(self.outcomes) / total if total else 0.0


def pair(classifications: list[Classification], labels: list[Label]) -> PairReport:
    """Join on product_id.

    Duplicate ids in either input raise. A second classification for the same product means
    either the batch ran twice or two providers' output got concatenated, and both make the
    resulting measurement describe nothing in particular -- silently keeping the last one
    would produce a plausible report over data whose provenance is unknown.
    """
    _reject_duplicates([item.product_id for item in classifications], "classifications")
    _reject_duplicates([item.product_id for item in labels], "labels")

    by_label = {item.product_id: item for item in labels}
    report = PairReport()
    for classification in classifications:
        label = by_label.get(classification.product_id)
        if label is None:
            report.unlabelled.append(classification.product_id)
            continue
        report.outcomes.append(Outcome(classification=classification, label=label))

    classified = {item.product_id for item in classifications}
    report.unclassified = sorted(by_label.keys() - classified)
    return report


def _reject_duplicates(ids: list[str], what: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in ids:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    if duplicates:
        raise ValueError(
            f"{what} contain duplicate product ids {sorted(duplicates)}; a measurement over "
            "the same product twice weights it twice"
        )
