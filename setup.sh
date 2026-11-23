#!/usr/bin/env bash
set -euo pipefail

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${BLUE}==>${NC} $1"; }
ok()   { echo -e "${GREEN} ok ${NC} $1"; }
warn() { echo -e "${YELLOW}warn${NC} $1"; }
fail() { echo -e "${RED}fail${NC} $1"; exit 1; }

WITH_CHART=0
for arg in "$@"; do
    case "$arg" in
        --chart|--all) WITH_CHART=1 ;;
        -h|--help)
            cat <<'USAGE'
Usage: bash setup.sh [options]

  --chart   also install matplotlib, for PNG reliability diagrams
  --all     everything
  -h        this message

Everything except the PNG works without extras, and nothing here needs credentials or a
network. The cloud providers (bedrock, vertex) are installable separately:

  pip install -e '.[bedrock]'      pip install -e '.[vertex]'
USAGE
            exit 0 ;;
        *) fail "unknown option: $arg (try --help)" ;;
    esac
done

info "Checking Python"
command -v python3 >/dev/null 2>&1 || fail "python3 not found (macOS: brew install python@3.12)"
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$(python3 -c 'import sys; print(1 if sys.version_info >= (3,11) else 0)')" != "1" ]; then
    fail "Python 3.11+ required (StrEnum, typing.Self). Found $PY_VER."
fi
ok "Python $PY_VER"

info "Creating virtual environment"
[ -d .venv ] || python3 -m venv .venv || fail "could not create .venv (Debian: apt install python3-venv)"
ok ".venv ready"

EXTRAS="dev"
[ "$WITH_CHART" = "1" ] && EXTRAS="$EXTRAS,chart"

info "Installing dependencies [$EXTRAS]"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e ".[$EXTRAS]" || fail "pip install failed"
ok "dependencies installed"

info "Verifying imports"
.venv/bin/python - <<'PYCHECK' || fail "core imports failed"
import click                                                       # noqa: F401
from pydantic_settings import BaseSettings                         # noqa: F401
from ai_classifier.core.calibrate import calibrate                 # noqa: F401
from ai_classifier.core.batch import run_batch                     # noqa: F401
from ai_classifier.providers import build                          # noqa: F401
from ai_classifier.render.text import render                       # noqa: F401
PYCHECK
ok "imports resolve"

info "Checking that a metric with an empty denominator reports nothing, not zero"
.venv/bin/python - <<'PYCHECK' || fail "the empty-denominator inversion is back"
from ai_classifier.core.calibrate import (
    calibrate, calibrate_by_modality, blind_spots, expected_calibration_error,
)
from ai_classifier.models import Classification, Label, Modality, Outcome, Verdict


def declined(index, modality=Modality.IMAGE_ONLY):
    pid = f"X-{index}"
    return Outcome(
        classification=Classification(
            product_id=pid, verdict=Verdict.NOT_ENOUGH_INFO, confidence=0.0,
            modality=modality, provider="probe",
        ),
        label=Label(product_id=pid, verdict=Verdict.TRUE_POSITIVE),
    )


def answered(index, correct, confidence=0.95, modality=Modality.TEXT_ONLY):
    pid = f"A-{modality.value}-{index}"
    return Outcome(
        classification=Classification(
            product_id=pid, verdict=Verdict.TRUE_POSITIVE, confidence=confidence,
            modality=modality, provider="probe",
        ),
        label=Label(
            product_id=pid,
            verdict=Verdict.TRUE_POSITIVE if correct else Verdict.FALSE_POSITIVE,
        ),
    )

# ECE is a DISTANCE, so 0.0 is the best attainable score. Returning it for a subgroup
# nobody answered does not merely lose information, it awards a perfect result.
assert expected_calibration_error([]) is None, "no bins must not score 0.0"

empty = calibrate([declined(i) for i in range(50)])
assert empty.ece is None, f"ece should be None, got {empty.ece}"
assert empty.accuracy is None, f"accuracy should be None, got {empty.accuracy}"
assert not empty.measurable
print("     50 abstentions -> ece None, accuracy None (0.0 would be a perfect score)")

# The two headline metrics used to point in OPPOSITE directions on the same subgroup:
# ECE 0.0 read as perfect, accuracy 0.0 read as total failure. Both were artefacts of
# the same empty denominator.
mixed = [answered(i, i < 21, modality=Modality.TEXT_ONLY) for i in range(30)]
mixed += [declined(i, Modality.IMAGE_ONLY) for i in range(43)]
reports = calibrate_by_modality(mixed)
assert reports[Modality.IMAGE_ONLY].ece is None
assert reports[Modality.TEXT_ONLY].ece is not None

# The strongest guarantee available: ranking cannot silently put a blind subgroup first.
try:
    sorted(reports.values(), key=lambda item: item.ece)
    raise AssertionError("sorting by ece succeeded; a blind subgroup could rank as best")
except TypeError:
    pass
print("     ranking subgroups by ece now RAISES instead of ranking a blind spot first")

spots = blind_spots(reports)
assert [item.subgroup for item in spots] == ["image_only"], spots
assert "NONE answered" in spots[0].describe()
print(f"     blind spot named explicitly: {spots[0].subgroup} (43 items, 0 answered)")
PYCHECK
ok "unmeasurable means unmeasurable"

info "Checking that abstaining cannot improve a calibration score"
.venv/bin/python - <<'PYCHECK' || fail "abstention still moves the ECE"
from ai_classifier.core.calibrate import calibrate
from ai_classifier.models import Classification, Label, Modality, Outcome, Verdict

def answered(index, correct):
    pid = f"A-{index}"
    return Outcome(
        classification=Classification(
            product_id=pid, verdict=Verdict.TRUE_POSITIVE, confidence=0.95,
            modality=Modality.TEXT_ONLY, provider="probe",
        ),
        label=Label(
            product_id=pid,
            verdict=Verdict.TRUE_POSITIVE if correct else Verdict.FALSE_POSITIVE,
        ),
    )

def declined(index):
    pid = f"D-{index}"
    return Outcome(
        classification=Classification(
            product_id=pid, verdict=Verdict.NOT_ENOUGH_INFO, confidence=0.0,
            modality=Modality.IMAGE_ONLY, provider="probe",
        ),
        label=Label(product_id=pid, verdict=Verdict.TRUE_POSITIVE),
    )

base = [answered(i, i < 28) for i in range(40)]
before = calibrate(base)
after = calibrate(base + [declined(i) for i in range(200)])
assert before.ece == after.ece, f"{before.ece} != {after.ece}"
assert sum(b.count for b in after.bins) == 40, "abstentions leaked into a bin"
print(f"     adding 200 abstentions left ece at {before.ece:.4f} and the bins at 40")

# An abstention asserted confidently is incoherent and must be refused, not repaired.
try:
    Classification(
        product_id="Z", verdict=Verdict.NOT_ENOUGH_INFO, confidence=0.97,
        modality=Modality.BOTH, provider="probe",
    )
    raise AssertionError("a 97%-confident abstention was accepted")
except ValueError:
    print("     'I cannot tell, and I am 97% sure' is refused at construction")
PYCHECK
ok "abstention is neither rewarded nor punished"

info "Checking that a negation does not fire the term it negates"
.venv/bin/python - <<'PYCHECK' || fail "the substring collision is back"
from ai_classifier.models import Product, Verdict
from ai_classifier.providers import build
from ai_classifier.providers.offline import CONTRADICTING, SUPPORTING

# "flammable" is a SUBSTRING of "non flammable". Scanning the two tables independently
# made a pure contradiction score on both sides and cancel to exactly zero, so the
# provider abstained on unambiguous evidence -- which reads as caution, not as a defect.
collisions = [
    (a, b)
    for a in SUPPORTING
    for b in CONTRADICTING
    if a != b and (a in b or b in a)
]
assert collisions, "the sample tables should still contain the collision this guards"
print(f"     {len(collisions)} cross-table substring collision(s): {collisions}")

provider = build("offline")
cases = [
    ("this item is non flammable", Verdict.FALSE_POSITIVE, "the negation wins"),
    ("contains flammable aerosol", Verdict.TRUE_POSITIVE, "both terms still count"),
    ("contains lithium and non flammable", Verdict.NOT_ENOUGH_INFO, "a genuine tie"),
]
for text, expected, why in cases:
    got = provider.classify(Product(product_id="P", attributes={"d": text}))
    assert got.verdict is expected, f"{text!r}: expected {expected}, got {got.verdict}"
    print(f"     {text:<38} {got.verdict.value:<16} {why}")
PYCHECK
ok "longest-match masking works"

info "Running the shipped catalogue through the pipeline"
.venv/bin/python - <<'PYCHECK' || fail "the shipped catalogue did not behave as documented"
from collections import Counter
from pathlib import Path

from ai_classifier.core.calibrate import blind_spots, calibrate, calibrate_by_modality
from ai_classifier.io import load_labels, load_products, pair
from ai_classifier.models import Modality, Verdict
from ai_classifier.providers import build

products = load_products(Path("data/products.csv"))
labels = load_labels(Path("data/labels.csv"))
assert not products.problems, products.problems
assert not labels.problems, labels.problems

provider = build("offline")
paired = pair([provider.classify(item) for item in products.products], labels.labels)
print(f"     {len(products.products)} products, {len(labels.labels)} usable labels, "
      f"{len(labels.undecided_labels)} undecided (excluded)")
print(f"     {len(paired.outcomes)} scored, {len(paired.unlabelled)} unlabelled")

report = calibrate(paired.outcomes)
assert report.measurable
print(f"     accuracy {report.accuracy:.1%}, ece {report.ece:.4f}, {report.direction}")

# Every modality and every verdict must be exercised, or the assertions below are vacuous.
mods = {item.classification.modality for item in paired.outcomes}
verdicts = {item.classification.verdict for item in paired.outcomes}
assert mods == set(Modality), f"missing {set(Modality) - mods}"
assert verdicts == set(Verdict), f"missing {set(Verdict) - verdicts}"
print(f"     every modality and verdict exercised: {dict(Counter(m.value for m in mods))}")

by_modality = calibrate_by_modality(paired.outcomes)
assert sum(item.total for item in by_modality.values()) == len(paired.outcomes)
image_only = by_modality[Modality.IMAGE_ONLY]
assert image_only.total > 0 and image_only.answered == 0
assert image_only.ece is None
print(f"     image_only: {image_only.total} items, 0 answered, ece not measurable")

spots = {item.subgroup for item in blind_spots(by_modality)}
assert spots == {"image_only", "neither"}, spots
print(f"     blind spots reported: {sorted(spots)}")

# Order independence: two runs over the same data must agree exactly.
import random
for seed in range(8):
    shuffled = list(paired.outcomes)
    random.Random(seed).shuffle(shuffled)
    other = calibrate(shuffled)
    assert other.ece == report.ece
    assert [b.count for b in other.bins] == [b.count for b in report.bins]
print("     order-independent across 8 shuffles")
PYCHECK
ok "the catalogue behaves as documented"

info "Checking that a batch resumes without repeating work"
.venv/bin/python - <<'PYCHECK' || fail "checkpoint resume is broken"
import tempfile
from pathlib import Path

from ai_classifier.core.batch import run_batch
from ai_classifier.io import load_products
from ai_classifier.providers import build

products = load_products(Path("data/products.csv")).products
checkpoint = Path(tempfile.mkdtemp()) / "ck.jsonl"

first = run_batch(products[:100], build("offline"), checkpoint_path=checkpoint)
assert first.newly_classified == 100 and first.from_checkpoint == 0

resumed = run_batch(products, build("offline"), checkpoint_path=checkpoint)
assert resumed.from_checkpoint == 100, resumed.from_checkpoint
assert resumed.newly_classified == len(products) - 100
print(f"     100 reused, {resumed.newly_classified} newly run, {resumed.total} total")

# A truncated final line is the expected result of a crash mid-flush.
with checkpoint.open("a", encoding="utf-8") as handle:
    handle.write('{"product_id": "P-9", "verdi')
again = run_batch(products, build("offline"), checkpoint_path=checkpoint)
assert again.truncated_checkpoint_lines, "a truncated line was not detected"
assert again.total == len(products)
print(f"     a truncated line was reported (line {again.truncated_checkpoint_lines[0]}) "
      "and the run still completed")

# Resuming under a different provider would blend two providers' verdicts into one
# calibration measurement, which would describe neither.
try:
    class Other(type(build("offline"))):
        name = "vertex"
    run_batch(products[:1], Other(), checkpoint_path=checkpoint)
    raise AssertionError("a provider switch was allowed")
except ValueError as error:
    assert "blend" in str(error) or "written by" in str(error), error
    print("     switching provider mid-checkpoint is refused")
PYCHECK
ok "checkpoint and resume work"

info "Checking that the report never prints 0.0000 for something unmeasured"
.venv/bin/python - <<'PYCHECK' || fail "the renderer restated the metric bug"
from pathlib import Path

from ai_classifier.core.calibrate import calibrate, choose_threshold
from ai_classifier.io import load_labels, load_products, pair
from ai_classifier.providers import build
from ai_classifier.render import text

products = load_products(Path("data/products.csv")).products
labels = load_labels(Path("data/labels.csv")).labels
outcomes = pair([build("offline").classify(p) for p in products], labels).outcomes
rendered = text.render(calibrate(outcomes), outcomes, choose_threshold(outcomes))

for line in rendered.splitlines():
    if "image_only" in line and "%" in line:
        assert "n/a" in line, line
        assert "0.0000" not in line, line
        print(f"     {line.strip()}")
assert "NONE answered" in rendered
assert "blind spots" in rendered
print("     blind spots are named in the report body, not left as a blank cell")
PYCHECK
ok "the renderer tells the truth"

if [ "$WITH_CHART" = "1" ]; then
    info "Rendering a reliability diagram"
    .venv/bin/python - <<'PYCHECK' || fail "the chart could not be written"
import tempfile
from pathlib import Path

from ai_classifier.core.calibrate import calibrate
from ai_classifier.io import load_labels, load_products, pair
from ai_classifier.providers import build
from ai_classifier.render.chart import write_chart

products = load_products(Path("data/products.csv")).products
labels = load_labels(Path("data/labels.csv")).labels
outcomes = pair([build("offline").classify(p) for p in products], labels).outcomes
target = Path(tempfile.mkdtemp()) / "reliability.png"
written = write_chart(calibrate(outcomes), target)
assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
print(f"     wrote a {written.stat().st_size} byte PNG")
PYCHECK
    ok "charts work"
else
    warn "matplotlib not installed, so PNG diagrams are unavailable (bash setup.sh --chart)"
fi

info "Running the test suite"
.venv/bin/python -m pytest -q || fail "tests failed"
ok "tests pass"

cat <<'DONE'

Setup complete.

  source .venv/bin/activate

  plumbline doctor
  plumbline providers
  plumbline measure -i data/products.csv -l data/labels.csv
  plumbline stability -i data/products.csv -l data/labels.csv
  plumbline classify -i data/products.csv -c out/ck.jsonl
  plumbline measure -i data/products.csv -l data/labels.csv -o out/

`measure` exits non-zero when a subgroup could not be measured at all, or when no
confidence threshold reaches the target accuracy. Both mean "do not automate this yet",
so it composes into a pipeline step that gates on evidence rather than on optimism.

ECE and accuracy print as n/a, never as 0.0000, when nothing was answered. ECE is a
distance: 0.0 is the BEST possible score, so printing it for a subgroup nobody answered
would rank a blind spot as the best-calibrated segment. It did, before this was fixed.
DONE
