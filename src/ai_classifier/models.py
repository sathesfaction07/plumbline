"""Domain types.

The vocabulary matters more than usual here. A classifier that can say "I don't know" is
a fundamentally different object from one that cannot, and almost every measurement in
`core/calibrate.py` has to decide what to do with that third answer. Getting the types
right is what makes those decisions explicit instead of accidental.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class Verdict(StrEnum):
    """What a classification concluded.

    TRUE_POSITIVE and FALSE_POSITIVE are answers. NOT_ENOUGH_INFO is an abstention, and
    it is deliberately in the same enum rather than being modelled as `verdict=None`:
    "I looked and could not tell" is a real, reportable conclusion, and a reviewer needs
    to see it. But it is NOT an answer, and treating it as one silently corrupts every
    accuracy and calibration number downstream -- see `is_answer`.
    """

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    NOT_ENOUGH_INFO = "not_enough_info"

    @property
    def is_answer(self) -> bool:
        """Whether this verdict commits to a conclusion.

        Every aggregate metric has to ask this question. Accuracy over a set that
        includes abstentions is not accuracy -- it conflates "got it wrong" with
        "declined to say", which are different failures with different remedies.
        """
        return self is not Verdict.NOT_ENOUGH_INFO


class Modality(StrEnum):
    """Which evidence a product actually carries.

    The spec's hard case is IMAGE_ONLY: a listing with pictures and no usable text. A
    text-only classifier cannot see it at all, so it either abstains on every one of them
    or -- worse -- guesses from the title. Naming the modality makes the routing decision
    inspectable rather than buried in a truthiness check on a description field.
    """

    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    BOTH = "both"
    NEITHER = "neither"

    @property
    def has_text(self) -> bool:
        return self in (Modality.TEXT_ONLY, Modality.BOTH)

    @property
    def has_image(self) -> bool:
        return self in (Modality.IMAGE_ONLY, Modality.BOTH)


class Product(BaseModel):
    """An item to classify.

    `attributes` is a free-form mapping because feeds disagree about which fields exist;
    `images` is a list of references (paths or URLs) that a provider may or may not be
    able to read. Neither is required, because NEITHER is a real modality and a pipeline
    that crashes on an empty listing is useless for exactly the inputs that need review.
    """

    product_id: str
    attributes: dict[str, str] = Field(default_factory=dict)
    images: list[str] = Field(default_factory=list)

    @field_validator("attributes")
    @classmethod
    def _drop_blank_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        """Blank strings are not text.

        A feed that emits `description: ""` for a product with no description would
        otherwise present as TEXT_ONLY, so the pipeline would send it to a text
        classifier that has nothing to read. That is the image-only failure mode arriving
        by a side door, which is why this is a validator and not a caller's problem.
        """
        return {key: value for key, value in value.items() if value and value.strip()}

    @property
    def modality(self) -> Modality:
        has_text = bool(self.attributes)
        has_image = bool(self.images)
        if has_text and has_image:
            return Modality.BOTH
        if has_text:
            return Modality.TEXT_ONLY
        if has_image:
            return Modality.IMAGE_ONLY
        return Modality.NEITHER

    def text_blob(self) -> str:
        """Attributes flattened for a text prompt, in sorted key order.

        Sorted so that two runs over the same product build the same prompt. Dict order
        follows insertion, which follows column order in the feed, which is not stable
        across exports -- and an unstable prompt makes a cached or replayed
        classification impossible to compare.
        """
        return "\n".join(f"{key}: {self.attributes[key]}" for key in sorted(self.attributes))


class Classification(BaseModel):
    """One provider's conclusion about one product.

    `confidence` is what the provider CLAIMS. Whether that claim means anything is the
    question `core/calibrate.py` exists to answer, and the naming is deliberate: this
    field is evidence about the provider, not a fact about the product.
    """

    product_id: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    modality: Modality
    provider: str
    rationale: str = ""
    latency_ms: int = 0

    @model_validator(mode="after")
    def _abstention_must_not_be_confident(self) -> Self:
        """An abstention asserted with high confidence is a contradiction.

        "I cannot tell, and I am 97% sure of my answer" is not a coherent output, but a
        provider can easily emit it -- an LLM asked for a verdict and a confidence will
        happily produce NOT_ENOUGH_INFO alongside the 0.95 it puts on everything. Left
        alone it poisons calibration from both ends: it inflates the high-confidence
        bins with items that were never answers, and it hides the abstention from any
        reviewer filtering on low confidence.

        Refused at construction rather than repaired, because the sensible repair
        (rewrite the confidence) invents a number the provider never reported.
        """
        if self.verdict is Verdict.NOT_ENOUGH_INFO and self.confidence > 0.5:
            raise ValueError(
                f"{self.product_id}: verdict is not_enough_info but confidence is "
                f"{self.confidence:.2f}. An abstention cannot be held confidently; a "
                "provider reporting both is misreporting one of them."
            )
        return self


class Label(BaseModel):
    """A human's verdict on a product: the ground truth calibration is measured against.

    `source` and `labelled_at` are carried because a calibration report is only as good
    as its labels, and "which reviewer, when" is the first question anyone asks when a
    measurement is surprising.
    """

    product_id: str
    verdict: Verdict
    source: str = "human"
    labelled_at: datetime | None = None

    @field_validator("verdict")
    @classmethod
    def _ground_truth_must_be_decided(cls, value: Verdict) -> Verdict:
        """A label of NOT_ENOUGH_INFO is not ground truth.

        A reviewer who could not decide has not produced a target to measure against.
        Admitting these would make "the classifier abstained and the human also
        abstained" score as a correct answer, which would let a provider that abstains
        on everything look accurate.

        Such rows are legitimate data -- they mark products needing better evidence --
        but they belong in a queue, not in a scoring set, so `io.load_labels` reports
        them as skipped rather than dropping them silently.
        """
        if value is Verdict.NOT_ENOUGH_INFO:
            raise ValueError(
                "a label of not_enough_info is not ground truth: there is nothing to "
                "measure against. Keep the row for review, exclude it from scoring."
            )
        return value


class Outcome(BaseModel):
    """A classification paired with its label.

    Deliberately NOT a boolean `correct` field. Three things can happen and they need
    different responses: the classifier answered and was right, it answered and was
    wrong, or it declined. Collapsing the third into "wrong" is the single most common
    way an abstaining classifier gets mismeasured.
    """

    classification: Classification
    label: Label

    @model_validator(mode="after")
    def _must_describe_the_same_product(self) -> Self:
        if self.classification.product_id != self.label.product_id:
            raise ValueError(
                f"outcome pairs {self.classification.product_id} with a label for "
                f"{self.label.product_id}"
            )
        return self

    @property
    def product_id(self) -> str:
        return self.classification.product_id

    @property
    def confidence(self) -> float:
        return self.classification.confidence

    @property
    def answered(self) -> bool:
        return self.classification.verdict.is_answer

    @property
    def correct(self) -> bool | None:
        """True, False, or None for an abstention.

        The `None` is the point. A caller writing `if outcome.correct:` treats an
        abstention as wrong; a caller writing `if outcome.correct is False:` does not.
        The three-valued return makes the distinction impossible to ignore by accident,
        because the wrong reading is at least visibly a choice.
        """
        if not self.answered:
            return None
        return self.classification.verdict is self.label.verdict
