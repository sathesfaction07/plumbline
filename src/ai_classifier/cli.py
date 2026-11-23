"""Command line interface.

Exit codes carry meaning, because these commands are meant to sit in a pipeline:

  measure   1 if any subgroup is a blind spot, or if the target accuracy is unreachable
  classify  1 if any product failed
  doctor    1 if the configured inputs cannot be read

`measure` failing on a blind spot is the deliberate consequence of the defect this project
exists to fix. A subgroup nobody answered used to score a perfect ECE; now it stops the
build, because "we measured nothing here" is a result somebody has to see.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from ai_classifier.config import Settings, load_settings
from ai_classifier.core.batch import run_batch
from ai_classifier.core.calibrate import (
    BinScheme,
    blind_spots,
    calibrate,
    calibrate_by_modality,
    choose_threshold,
)
from ai_classifier.core.feedback import threshold_stability
from ai_classifier.io import load_labels, load_products, pair
from ai_classifier.models import Classification
from ai_classifier.providers import NAMES, build
from ai_classifier.render import export, text


def _err(message: str) -> None:
    """Diagnostics to stderr, so stdout stays pipeable."""
    click.echo(message, err=True)


def _load(settings: Settings, products: Path | None, labels: Path | None):
    product_report = load_products(products or settings.products)
    label_report = load_labels(labels or settings.labels)
    for problem in product_report.problems + label_report.problems:
        _err(f"  problem: {problem}")
    return product_report, label_report


@click.group()
@click.version_option("0.1.0", prog_name="plumbline")
def main() -> None:
    """Multimodal product classification with measured confidence calibration."""


@main.command()
@click.option("-i", "--products", type=click.Path(exists=True, path_type=Path))
@click.option("-p", "--provider", type=click.Choice(NAMES), default=None)
@click.option("-c", "--checkpoint", type=click.Path(path_type=Path),
              help="resume from and append to this JSON Lines file")
@click.option("-o", "--out", type=click.Path(path_type=Path),
              help="write classifications here as JSON Lines")
def classify(products: Path | None, provider: str | None, checkpoint: Path | None,
             out: Path | None) -> None:
    """Classify a catalogue, resuming from a checkpoint if given."""
    settings = load_settings()
    product_report = load_products(products or settings.products)
    for problem in product_report.problems:
        _err(f"  problem: {problem}")

    engine = build(provider or settings.provider)
    result = run_batch(product_report.products, engine, checkpoint_path=checkpoint)

    click.echo(f"provider          {engine.name}")
    click.echo(f"classified        {result.total}")
    click.echo(f"  from checkpoint {result.from_checkpoint}")
    click.echo(f"  newly run       {result.newly_classified}")
    if result.truncated_checkpoint_lines:
        _err(f"  checkpoint had unreadable line(s): {result.truncated_checkpoint_lines}")
    for product_id, why in result.failed.items():
        _err(f"  failed {product_id}: {why}")

    if out:
        export.write_jsonl(
            [json.loads(item.model_dump_json()) for item in result.classifications], out
        )
        click.echo(f"wrote             {out}")

    sys.exit(1 if result.failed else 0)


@main.command()
@click.option("-i", "--products", type=click.Path(exists=True, path_type=Path))
@click.option("-l", "--labels", type=click.Path(exists=True, path_type=Path))
@click.option("-p", "--provider", type=click.Choice(NAMES), default=None)
@click.option("-b", "--bins", "bin_count", type=int, default=None)
@click.option("-s", "--scheme", type=click.Choice([item.value for item in BinScheme]),
              default=BinScheme.EQUAL_WIDTH.value)
@click.option("-t", "--target", type=float, default=None,
              help="accuracy the auto-accept threshold must reach")
@click.option("-o", "--out", type=click.Path(path_type=Path), help="also export artefacts here")
@click.option("--no-confusion", is_flag=True)
def measure(products: Path | None, labels: Path | None, provider: str | None,
            bin_count: int | None, scheme: str, target: float | None,
            out: Path | None, no_confusion: bool) -> None:
    """Classify, compare against labels, and report calibration.

    Exits non-zero when a subgroup could not be measured at all or when no threshold
    reaches the target, because both mean the answer is "do not automate this yet".
    """
    settings = load_settings()
    product_report, label_report = _load(settings, products, labels)
    if label_report.undecided_labels:
        _err(
            f"  {len(label_report.undecided_labels)} label(s) were not_enough_info and are "
            "excluded from scoring; they are review items, not ground truth"
        )

    engine = build(provider or settings.provider)
    classifications = [engine.classify(item) for item in product_report.products]
    paired = pair(classifications, label_report.labels)
    if paired.unlabelled:
        _err(f"  {len(paired.unlabelled)} classified product(s) have no label; not scored")
    if paired.unclassified:
        _err(f"  {len(paired.unclassified)} labelled product(s) were not classified")

    chosen = BinScheme(scheme)
    report = calibrate(
        paired.outcomes, bin_count=bin_count or settings.bin_count, scheme=chosen
    )
    threshold = choose_threshold(
        paired.outcomes, target_accuracy=target or settings.target_accuracy
    )
    click.echo(text.render(report, paired.outcomes, threshold,
                           show_confusion=not no_confusion), nl=False)

    spots = blind_spots(
        calibrate_by_modality(paired.outcomes, bin_count=report.bin_count, scheme=chosen)
    )
    if out:
        for path in export.export_all(report, paired.outcomes, threshold, out):
            click.echo(f"wrote {path}")

    sys.exit(1 if spots or not threshold.reachable else 0)


@main.command()
@click.option("-i", "--products", type=click.Path(exists=True, path_type=Path))
@click.option("-l", "--labels", type=click.Path(exists=True, path_type=Path))
@click.option("-p", "--provider", type=click.Choice(NAMES), default=None)
@click.option("-t", "--target", type=float, default=None)
@click.option("--steps", type=int, default=8)
def stability(products: Path | None, labels: Path | None, provider: str | None,
              target: float | None, steps: int) -> None:
    """Show whether the threshold recommendation has settled as labels accumulated."""
    settings = load_settings()
    product_report, label_report = _load(settings, products, labels)
    engine = build(provider or settings.provider)
    outcomes = pair(
        [engine.classify(item) for item in product_report.products], label_report.labels
    ).outcomes

    report = threshold_stability(
        outcomes, target_accuracy=target or settings.target_accuracy, steps=steps
    )
    click.echo("labels used   threshold   accuracy at that point")
    for point in report.points:
        shown = "none reachable" if point.threshold is None else f"{point.threshold:.2f}"
        click.echo(f"{point.labels_used:>11}   {shown:>9}   {point.achieved_accuracy:.1%}")
    if report.stable:
        click.echo(f"\nsettled after {report.settled_after} label(s)")
    else:
        click.echo(
            "\nstill moving at the last measurement: there is not yet enough ground "
            "truth to act on this recommendation"
        )


@main.command()
@click.argument("outcomes_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--out", type=click.Path(path_type=Path), default=Path("reliability.png"))
def chart(outcomes_file: Path, out: Path) -> None:
    """Render a reliability diagram as PNG from exported classifications plus labels."""
    from ai_classifier.core.batch import iter_jsonl  # noqa: PLC0415
    from ai_classifier.render.chart import write_chart  # noqa: PLC0415

    settings = load_settings()
    classifications = [Classification.model_validate(row) for row in iter_jsonl(outcomes_file)]
    outcomes = pair(classifications, load_labels(settings.labels).labels).outcomes
    written = write_chart(calibrate(outcomes), out, title="all modalities")
    click.echo(f"wrote {written}")


@main.command()
def providers() -> None:
    """List providers and what each can read."""
    click.echo("name      reads              needs")
    click.echo("offline   text only          nothing (default)")
    click.echo("bedrock   text and images    boto3, AWS credentials")
    click.echo("vertex    text and images    google-cloud-aiplatform, GCP credentials")
    click.echo("")
    click.echo("A text-only provider abstains on every image-only listing. That shows up as")
    click.echo("a blind spot rather than as a good calibration score -- see `measure`.")


@main.command()
@click.option("-i", "--products", type=click.Path(path_type=Path))
@click.option("-l", "--labels", type=click.Path(path_type=Path))
def doctor(products: Path | None, labels: Path | None) -> None:
    """Check configuration and whether the inputs can be read."""
    settings = load_settings()
    product_path = products or settings.products
    label_path = labels or settings.labels
    ok = True

    click.echo("configuration")
    click.echo(f"  products      {product_path}")
    click.echo(f"  labels        {label_path}")
    click.echo(f"  provider      {settings.provider}")
    click.echo(f"  bins          {settings.bin_count}")
    click.echo(f"  target        {settings.target_accuracy:.0%}")

    click.echo("\ninputs")
    for path, loader, what in ((product_path, load_products, "products"),
                               (label_path, load_labels, "labels")):
        if not path.exists():
            click.echo(f"  {what:<12} MISSING {path}")
            ok = False
            continue
        try:
            report = loader(path)
        except (OSError, ValueError) as error:
            click.echo(f"  {what:<12} UNREADABLE {error}")
            ok = False
            continue
        count = len(report.products or report.labels)
        extra = ""
        if report.undecided_labels:
            extra = f", {len(report.undecided_labels)} undecided (excluded from scoring)"
        click.echo(f"  {what:<12} {count} row(s), {len(report.problems)} problem(s){extra}")

    click.echo("\nreading the report")
    click.echo("  ECE and accuracy print as n/a when nothing was answered. That is not a")
    click.echo("  formatting choice: 0.0000 is the BEST possible ECE, so printing it for a")
    click.echo("  subgroup nobody answered would rank a blind spot as the best result.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
