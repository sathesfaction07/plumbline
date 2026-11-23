"""The offline provider: text signals, no network, no model.

This is the spec's "local, for offline use". It reads keyword signals out of a product's
attributes and weighs those supporting the flag against those contradicting it.

It is deliberately the DEFAULT, for two reasons. It runs in CI with no credentials, so the
whole pipeline is testable end to end. And it makes the image-only problem impossible to
paper over: this provider genuinely cannot see a picture, so a listing with no usable text
is a listing it has nothing to say about. A cloud vision provider makes that failure
invisible by answering anyway, which is a worse position to reason from.

The confidence curve here saturates high -- a clear signal margin produces 0.9+. That is
modelled on how real classifiers behave rather than chosen to be flattering, and it is the
behaviour `core/calibrate.py` exists to check.
"""

from __future__ import annotations

from ai_classifier.models import Classification, Modality, Product, Verdict
from ai_classifier.providers.base import Provider

# Signals supporting the flag (the item really is what it was flagged as) and signals
# contradicting it. Weights are coarse on purpose: three tiers, because a keyword list with
# finely tuned decimal weights implies a precision that a keyword list does not have.
SUPPORTING: dict[str, float] = {
    "lithium": 3.0,
    "aerosol": 3.0,
    "flammable": 3.0,
    "pressurised": 2.0,
    "compressed gas": 3.0,
    "battery pack": 2.0,
    "sharp blade": 2.0,
    "fixed blade": 2.0,
    "magnet": 1.0,
    "small parts": 1.0,
    "choking": 2.0,
}

CONTRADICTING: dict[str, float] = {
    "battery free": 3.0,
    "no batteries": 3.0,
    "non flammable": 3.0,
    "water based": 2.0,
    "blunt": 2.0,
    "rounded tip": 2.0,
    "plush": 1.0,
    "fabric only": 2.0,
    "certified": 1.0,
    "age 12+": 1.0,
}


def _match_signals(blob: str) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Find supporting and contradicting terms, longest phrase first, masking as it goes.

    Scanning each table independently is the obvious implementation and it is wrong,
    because one table's term can be a SUBSTRING of the other's. Here `"flammable"` is
    inside `"non flammable"`, so a listing reading "non flammable" -- an unambiguous
    contradiction -- scored 3 supporting AND 3 contradicting and cancelled to exactly zero.
    The provider then abstained on clear evidence, which is invisible in output: an
    abstention looks like caution, not like a matching defect.

    Longest first, with each match blanked out so nothing shorter can match inside it. A
    negation therefore consumes the word it negates. Word boundaries alone would not fix
    this -- "flammable" sits on clean boundaries within "non flammable".
    """
    remaining = blob.lower()
    ordered = sorted(
        [(term, weight, True) for term, weight in SUPPORTING.items()]
        + [(term, weight, False) for term, weight in CONTRADICTING.items()],
        key=lambda entry: -len(entry[0]),
    )

    supporting: list[tuple[str, float]] = []
    contradicting: list[tuple[str, float]] = []
    for term, weight, is_supporting in ordered:
        index = remaining.find(term)
        if index < 0:
            continue
        # Blank the span rather than deleting it, so later offsets stay meaningful and a
        # term appearing twice is still only counted once.
        remaining = remaining[:index] + " " * len(term) + remaining[index + len(term) :]
        (supporting if is_supporting else contradicting).append((term, weight))

    return supporting, contradicting


def _confidence(margin: float, total: float) -> float:
    """Map signal margin onto a stated confidence.

    Saturating: a decisive margin lands near 0.95 rather than continuing to climb. Real
    classifiers behave this way, and the shape matters for what calibration then measures
    -- most of the mass ends up in the top two bins, which is why an equal-width binning
    leaves the middle of the range nearly empty.

    `total` damps the result when signals conflict: two supporting and two contradicting
    terms is a genuinely uncertain case even though both sides are strong, and a margin
    alone cannot tell that apart from a case with no signal at all.
    """
    if total == 0:
        return 0.0
    agreement = abs(margin) / total
    return round(0.55 + 0.40 * agreement, 3)


class OfflineProvider(Provider):
    """Weighs text signals. Abstains when it has no text to read."""

    name = "offline"

    def supports(self, modality: Modality) -> bool:
        return modality.has_text

    def classify(self, product: Product) -> Classification:
        if not self.supports(product.modality):
            return self.unable(
                product,
                f"no readable text ({product.modality.value}); this provider cannot see images",
            )

        supporting, contradicting = _match_signals(product.text_blob())

        support_weight = sum(weight for _, weight in supporting)
        contradict_weight = sum(weight for _, weight in contradicting)
        margin = support_weight - contradict_weight
        total = support_weight + contradict_weight

        if total == 0:
            return self.unable(product, "text present but no known signal appears in it")

        if margin == 0:
            # Signals cancel exactly. This is the ambiguous-product case from the spec, and
            # abstaining is the honest answer -- picking a side on a tie would produce a
            # verdict whose confidence could not be anything but arbitrary.
            return self.unable(
                product,
                f"signals cancel: {support_weight:.0f} supporting vs "
                f"{contradict_weight:.0f} contradicting",
            )

        verdict = Verdict.TRUE_POSITIVE if margin > 0 else Verdict.FALSE_POSITIVE
        winning = supporting if margin > 0 else contradicting
        terms = ", ".join(term for term, _ in sorted(winning, key=lambda pair: -pair[1]))

        return Classification(
            product_id=product.product_id,
            verdict=verdict,
            confidence=_confidence(margin, total),
            modality=product.modality,
            provider=self.name,
            rationale=f"{terms} (margin {margin:+.0f} of {total:.0f})",
        )
