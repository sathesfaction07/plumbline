"""Shared fixtures and builders.

The shipped catalogue is loaded once per session and used by the property tests, so editing
`data/` and breaking an invariant fails the build rather than going unnoticed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ai_classifier.io import load_labels, load_products, pair
from ai_classifier.models import Classification, Label, Modality, Outcome, Product, Verdict
from ai_classifier.providers import build

requires_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib is an optional extra: pip install -e '.[chart]'",
)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def products():
    report = load_products(DATA / "products.csv")
    assert not report.problems, report.problems
    return report.products


@pytest.fixture(scope="session")
def labels():
    return load_labels(DATA / "labels.csv")


@pytest.fixture(scope="session")
def outcomes(products, labels):
    engine = build("offline")
    return pair([engine.classify(item) for item in products], labels.labels).outcomes


def classification(
    product_id: str = "P-1",
    verdict: Verdict = Verdict.TRUE_POSITIVE,
    confidence: float = 0.9,
    modality: Modality = Modality.TEXT_ONLY,
    provider: str = "offline",
) -> Classification:
    return Classification(
        product_id=product_id,
        verdict=verdict,
        confidence=confidence,
        modality=modality,
        provider=provider,
    )


def outcome(
    product_id: str = "P-1",
    predicted: Verdict = Verdict.TRUE_POSITIVE,
    actual: Verdict = Verdict.TRUE_POSITIVE,
    confidence: float = 0.9,
    modality: Modality = Modality.TEXT_ONLY,
) -> Outcome:
    return Outcome(
        classification=classification(product_id, predicted, confidence, modality),
        label=Label(product_id=product_id, verdict=actual),
    )


def many(
    count: int,
    *,
    confidence: float,
    accuracy: float,
    modality: Modality = Modality.TEXT_ONLY,
    prefix: str = "P",
) -> list[Outcome]:
    """`count` outcomes at one confidence, of which `accuracy` are correct.

    Used to build a population whose true accuracy is known exactly, so a calibration gap
    can be asserted against arithmetic rather than against a previous run's output.
    """
    right = round(count * accuracy)
    result = []
    for index in range(count):
        actual = Verdict.TRUE_POSITIVE if index < right else Verdict.FALSE_POSITIVE
        result.append(
            outcome(
                product_id=f"{prefix}-{modality.value}-{confidence}-{index}",
                predicted=Verdict.TRUE_POSITIVE,
                actual=actual,
                confidence=confidence,
                modality=modality,
            )
        )
    return result


def abstentions(count: int, *, modality: Modality = Modality.IMAGE_ONLY, prefix: str = "A"):
    return [
        outcome(
            product_id=f"{prefix}-{index}",
            predicted=Verdict.NOT_ENOUGH_INFO,
            actual=Verdict.TRUE_POSITIVE,
            confidence=0.0,
            modality=modality,
        )
        for index in range(count)
    ]


def product(product_id: str = "P-1", *, text: str = "", images: int = 0) -> Product:
    return Product(
        product_id=product_id,
        attributes={"description": text} if text else {},
        images=[f"img/{product_id}-{i}.jpg" for i in range(images)],
    )
