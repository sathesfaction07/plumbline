"""Reliability diagram as a PNG. Optional: needs matplotlib.

Imported lazily inside the function so `import ai_classifier` works without the extra. The
text version in `text.py` is the primary renderer because it can be asserted in CI; this one
is easier to put in front of a person.

A subgroup with no answers draws NOTHING and is labelled "not measured" in the title rather
than being drawn as a flat line at zero. A flat line on a reliability diagram sits exactly
on the y=0 axis, which a reader interprets as "always wrong" -- the opposite of the truth,
and a third distinct misreading of the same empty denominator.
"""

from __future__ import annotations

from pathlib import Path

from ai_classifier.core.calibrate import CalibrationReport


def write_chart(report: CalibrationReport, path: Path, *, title: str | None = None) -> Path:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "charts need matplotlib: pip install -e '.[chart]'"
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6, 6))
    axes.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey",
              label="perfect calibration")

    if report.bins:
        xs = [item.mean_confidence for item in report.bins]
        ys = [item.observed_accuracy for item in report.bins]
        sizes = [max(20, min(400, item.count * 3)) for item in report.bins]
        axes.plot(xs, ys, marker="", linewidth=1.5, color="tab:blue")
        axes.scatter(xs, ys, s=sizes, alpha=0.75, color="tab:blue", label="observed")
        for item in report.bins:
            if not item.is_reliable:
                axes.annotate("thin", (item.mean_confidence, item.observed_accuracy),
                              textcoords="offset points", xytext=(6, -10), fontsize=7,
                              color="tab:red")
        suffix = f"ECE {report.ece:.4f}" if report.ece is not None else "ECE not measurable"
    else:
        axes.text(0.5, 0.5, "nothing was answered\nno curve exists", ha="center",
                  va="center", fontsize=11, color="tab:red")
        suffix = "not measured"

    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_xlabel("claimed confidence")
    axes.set_ylabel("observed accuracy")
    axes.set_title(f"{title or report.subgroup} - {suffix}")
    axes.legend(loc="upper left", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path
