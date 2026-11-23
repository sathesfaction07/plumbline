"""The human feedback loop.

Routed items come back with a reviewer's verdict. Those verdicts are labels, so the loop is
simply the label set growing -- and the interesting question is not how to merge them but
whether the recommendation they support is stable yet.

A threshold computed on fifty labels is a number, not a finding. `threshold_stability`
answers "would more labels have changed my advice", which is the question that decides
whether to act on it.
"""

from __future__ import annotations

from pydantic import BaseModel

from ai_classifier.core.calibrate import Threshold, choose_threshold
from ai_classifier.models import Label, Outcome, Verdict


class FeedbackReport(BaseModel):
    """What a round of human review changed about the label set."""

    added: list[str] = []
    unchanged: list[str] = []
    overturned: list[str] = []
    undecided: list[str] = []

    @property
    def total(self) -> int:
        return len(self.added) + len(self.unchanged) + len(self.overturned) + len(self.undecided)


def apply_feedback(
    existing: list[Label], incoming: list[Label]
) -> tuple[list[Label], FeedbackReport]:
    """Merge reviewer verdicts into the label set.

    A second reviewer contradicting the first is `overturned` and the NEW verdict wins,
    because feedback arriving later is normally a correction. But it is reported rather than
    absorbed: a rising overturn rate means the labels themselves are unreliable, and that
    invalidates every calibration number computed from them. Silently taking the newest
    would hide the one signal that tells you to stop trusting your own ground truth.
    """
    by_id = {item.product_id: item for item in existing}
    report = FeedbackReport()

    for item in incoming:
        if item.verdict is Verdict.NOT_ENOUGH_INFO:
            # Cannot occur via Label construction, which refuses it; kept because a caller
            # may build Labels another way and this must not become a silent pass-through.
            report.undecided.append(item.product_id)
            continue
        previous = by_id.get(item.product_id)
        if previous is None:
            report.added.append(item.product_id)
        elif previous.verdict is item.verdict:
            report.unchanged.append(item.product_id)
            continue
        else:
            report.overturned.append(item.product_id)
        by_id[item.product_id] = item

    merged = sorted(by_id.values(), key=lambda item: item.product_id)
    return merged, report


class StabilityPoint(BaseModel):
    """The recommendation as it stood after a given number of labels."""

    labels_used: int
    threshold: float | None
    achieved_accuracy: float


class StabilityReport(BaseModel):
    """How much the threshold moved as labels accumulated."""

    points: list[StabilityPoint]
    final: Threshold

    @property
    def settled_after(self) -> int | None:
        """Labels needed before the recommendation stopped moving, or None if it never did.

        Walks backwards to find the earliest point from which every later point agrees with
        the final answer. None means the threshold was still moving at the last measurement,
        which means there is not yet enough ground truth to act on -- a genuinely useful
        thing to be told before automating anything.
        """
        if not self.points:
            return None
        target = self.final.value
        settled = None
        for point in reversed(self.points):
            if point.threshold != target:
                break
            settled = point.labels_used
        return settled

    @property
    def stable(self) -> bool:
        return self.settled_after is not None


def threshold_stability(
    outcomes: list[Outcome],
    *,
    target_accuracy: float = 0.95,
    steps: int = 8,
) -> StabilityReport:
    """Recompute the threshold over growing prefixes of the outcome set.

    Prefixes in the given order, NOT random subsamples. The order labels arrive in is real
    information -- reviewers work through a queue, and if the queue is sorted by anything
    correlated with difficulty then early labels are not representative. Random subsampling
    would average that away and report a stability that the actual arrival order does not
    have.

    The cost is that this measures stability under one ordering rather than in general. It
    is the ordering you actually had, which is the one that produced the number you are
    deciding whether to trust.
    """
    if not outcomes:
        return StabilityReport(
            points=[], final=choose_threshold([], target_accuracy=target_accuracy)
        )

    points = []
    for step in range(1, steps + 1):
        cut = max(1, len(outcomes) * step // steps)
        prefix = outcomes[:cut]
        threshold = choose_threshold(prefix, target_accuracy=target_accuracy)
        points.append(
            StabilityPoint(
                labels_used=cut,
                threshold=threshold.value,
                achieved_accuracy=threshold.achieved_accuracy,
            )
        )

    return StabilityReport(
        points=points,
        final=choose_threshold(outcomes, target_accuracy=target_accuracy),
    )
