"""Batch classification with checkpoint and resume.

A run over a large catalogue against a paid vision API is expensive and slow, so it has to
survive interruption without repeating work. The checkpoint is an append-only JSON Lines
file: one classification per line, flushed as it is produced.

Append-only rather than a periodically rewritten snapshot. A snapshot has a window during
which the file is partially written, and a crash inside that window loses everything
already done -- which is the exact scenario the checkpoint exists for. Appending means the
worst case is one truncated final line, and a truncated line is detectable.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel

from ai_classifier.models import Classification, Product
from ai_classifier.providers.base import Provider


class Checkpoint:
    """An append-only record of completed classifications.

    Keyed on product_id alone, not on (product_id, provider). A resume that silently
    re-ran everything because the provider string changed would defeat the purpose, and a
    resume that mixed two providers' verdicts into one report would produce a calibration
    measurement of no single thing. `load` therefore reports which providers it saw and
    `run_batch` refuses to continue a run under a different one.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.done: dict[str, Classification] = {}
        self.truncated_lines: list[int] = []
        self.providers_seen: set[str] = set()

    def load(self) -> None:
        """Read an existing checkpoint, tolerating one truncated final line.

        A partially written last line is the expected result of a crash mid-flush, so it is
        recorded and skipped rather than raising. A malformed line ANYWHERE ELSE is a
        different matter -- that suggests the file was edited or interleaved by two writers
        -- but distinguishing them requires knowing the line was last, so the count is
        reported and the caller decides.
        """
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = Classification.model_validate_json(line)
            except ValueError:
                self.truncated_lines.append(number)
                continue
            self.done[record.product_id] = record
            self.providers_seen.add(record.provider)

    def append(self, classification: Classification) -> None:
        """Write one record and flush immediately.

        Opened and closed per record, which is slower than holding a handle and far more
        robust: the OS has the bytes before the next network call starts, so an interruption
        during that call cannot lose the result that preceded it. For a batch whose per-item
        cost is a vision API round trip, the file overhead is not measurable.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(classification.model_dump_json() + "\n")
        self.done[classification.product_id] = classification
        self.providers_seen.add(classification.provider)


class BatchResult(BaseModel):
    """What a batch run did, including what it skipped and why."""

    classifications: list[Classification]
    from_checkpoint: int
    newly_classified: int
    failed: dict[str, str] = {}
    truncated_checkpoint_lines: list[int] = []

    @property
    def total(self) -> int:
        return len(self.classifications)


def run_batch(
    products: Iterable[Product],
    provider: Provider,
    *,
    checkpoint_path: Path | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> BatchResult:
    """Classify products, resuming from a checkpoint when one exists.

    A provider raising on one product does not abort the run -- the failure is recorded
    against that product_id and the batch continues. Aborting would mean one malformed
    listing costs you the rest of the catalogue, and for a paid API that is real money.

    Failures are NOT checkpointed. A transient network error should be retried on the next
    run, and writing it to the checkpoint would make it permanent. The consequence is that
    a product failing deterministically is retried on every run, which is visible in the
    failure count rather than being silently absorbed.
    """
    items = list(products)
    checkpoint = Checkpoint(checkpoint_path) if checkpoint_path else None
    if checkpoint:
        checkpoint.load()
        if len(checkpoint.providers_seen) > 1:
            raise ValueError(
                f"checkpoint {checkpoint.path} mixes providers "
                f"{sorted(checkpoint.providers_seen)}; a calibration measurement over two "
                "providers describes neither. Use one checkpoint per provider."
            )
        if checkpoint.providers_seen and provider.name not in checkpoint.providers_seen:
            existing = next(iter(checkpoint.providers_seen))
            raise ValueError(
                f"checkpoint {checkpoint.path} was written by {existing!r} but this run "
                f"uses {provider.name!r}. Resuming would blend two providers' verdicts."
            )

    results: list[Classification] = []
    failed: dict[str, str] = {}
    reused = 0

    for index, product in enumerate(items, start=1):
        if checkpoint and product.product_id in checkpoint.done:
            results.append(checkpoint.done[product.product_id])
            reused += 1
            if on_progress:
                on_progress(product.product_id, index, len(items))
            continue

        try:
            classification = provider.classify(product)
        except Exception as error:  # noqa: BLE001 - one bad product must not end the run
            failed[product.product_id] = f"{type(error).__name__}: {error}"
            if on_progress:
                on_progress(product.product_id, index, len(items))
            continue

        results.append(classification)
        if checkpoint:
            checkpoint.append(classification)
        if on_progress:
            on_progress(product.product_id, index, len(items))

    return BatchResult(
        classifications=results,
        from_checkpoint=reused,
        newly_classified=len(results) - reused,
        failed=failed,
        truncated_checkpoint_lines=checkpoint.truncated_lines if checkpoint else [],
    )


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield objects from a JSON Lines file, skipping blanks."""
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
