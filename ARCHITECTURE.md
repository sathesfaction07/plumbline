# AIClassifier — Technical Architecture

## Overview

Classifies products as true positive / false positive / not enough info across three providers, then measures whether the stated confidence means anything and recommends what may safely be automated.

**Design principle:** a confidence is a *claim about the provider*, not a fact about the product. Nothing here treats a confidence as trustworthy until it has been checked against ground truth, and every metric that could be computed from an empty set refuses to produce a number instead.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          CLI (ai-classifier)                                  │
│                                                                              │
│  measure     classify -> pair with labels -> calibrate -> recommend           │
│              EXIT 1 if any subgroup is unmeasurable, OR no threshold          │
│                reaches the target. Both mean "do not automate this yet"       │
│  classify    batch with checkpoint/resume; exit 1 if any product failed       │
│  stability   has the recommendation settled as labels accumulated?            │
│  providers   the three backends and what each can read                        │
│  doctor      config, inputs, AND how to read n/a in the report                │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌─────────────────┐   ┌──────────────────────┐   ┌───────────────────────┐
│   PROVIDERS     │   │        CORE          │   │       RENDER          │
│                 │   │                      │   │                       │
│ base.py         │   │ calibrate.py         │   │ text.py               │
│  Provider ABC   │   │  THE SUBSTANTIVE     │   │  reliability diagram  │
│  supports()     │   │  PART                │   │  as characters, so a  │
│  classify()     │   │                      │   │  test can pin it      │
│  unable() ->    │   │  reliability_bins    │   │  n/a, NEVER 0.0000    │
│   an HONEST     │   │   equal_width or     │   │                       │
│   abstention,   │   │   equal_count        │   │ chart.py  OPTIONAL    │
│   never a       │   │   ABSTENTIONS        │   │  matplotlib, Agg      │
│   filtered-out  │   │   EXCLUDED           │   │  no curve when        │
│   silence       │   │                      │   │  nothing was answered │
│                 │   │  expected_calibration│   │  (a flat line at 0    │
│ offline.py      │   │   _error -> float|   │   │   reads "always       │
│  DEFAULT        │   │   None               │   │   wrong")             │
│  text signals,  │   │                      │   │                       │
│  no network,    │   │  calibrate_by_       │   │ export.py             │
│  no model       │   │   modality           │   │  JSON Lines, CSV,     │
│  longest-match  │   │   the aggregate      │   │  report.json          │
│  masking        │   │   CANNOT show a      │   │  null, NEVER 0        │
│                 │   │   blind subgroup     │   │  header even if empty │
│ bedrock.py      │   │                      │   └───────────────────────┘
│ vertex.py       │   │  blind_spots         │
│  deferred       │   │   names what was     │
│  imports;       │   │   NOT measured       │
│  parse_response │   │                      │
│  is module-     │   │  choose_threshold    │
│  level and      │   │   on OBSERVED        │
│  TESTED         │   │   accuracy, never    │
│                 │   │   on the claim       │
└─────────────────┘   │   value: float|None  │
                      │                      │
                      │ batch.py             │
                      │  append-only JSONL   │
                      │  checkpoint          │
                      │  one provider per    │
                      │  checkpoint, ENFORCED│
                      │  failures NOT saved  │
                      │                      │
                      │ feedback.py          │
                      │  merge reviewer      │
                      │  verdicts, report    │
                      │  OVERTURNS           │
                      │  threshold_stability │
                      │   over PREFIXES      │
                      └──────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │  models.py  io.py    │
                      │                      │
                      │  Verdict: TP FP NEI  │
                      │   is_answer          │
                      │  Modality: 4 values  │
                      │  Classification      │
                      │   refuses a CONFIDENT│
                      │   abstention         │
                      │  Label refuses NEI   │
                      │  Outcome.correct     │
                      │   -> bool | None     │
                      │                      │
                      │  pair() reports BOTH │
                      │   kinds of miss      │
                      └──────────────────────┘
```

---

## Components

### models.py — the vocabulary

`Verdict` has three values and `is_answer` names which two are answers. `NOT_ENOUGH_INFO` is an abstention: a real conclusion, not an error, and not an answer. Every aggregate downstream has to decide what to do with it, and ADR-001 records each decision.

`Modality` has four values because a product either has text, images, both, or neither. Naming it makes routing inspectable rather than buried in a truthiness check on a description field. A validator drops blank attributes, because a feed emitting `description: ""` would otherwise present as `TEXT_ONLY` and get routed to a text classifier with nothing to read — the image-only failure arriving by a side door.

`Classification` refuses an abstention above confidence 0.5. `Label` refuses a verdict of `not_enough_info` outright. `Outcome.correct` returns `True | False | None`.

### core/calibrate.py — the substance

Two binning schemes, offered because they disagree on the same data and a single ECE carries an unstated binning choice. `_bin_index_equal_width` clamps, because `int(1.0 * 10)` is 10 — one past the last bin — and a confidence of exactly 1.0 is not hypothetical.

`expected_calibration_error` is **population-weighted**, not a plain mean over bins. An unweighted mean lets a bin holding three samples count as much as one holding three hundred, so with equal-width binning the headline would be dominated by the sparsest bands — exactly where confidences are rare.

`choose_threshold` scans **observed** accuracy, never the stated confidence. Using the claim would assume the calibration this function exists to work around.

`calibrate_by_modality` exists because the aggregate cannot show that a provider is blind to a whole subgroup: a subgroup it never answered contributes nothing to the aggregate — not a bad number, no number at all.

### providers/ — three backends

`offline` is the default and needs nothing. It weighs keyword signals with **longest-match-first masking**, so a negation consumes the term it negates (see below). Its confidence curve saturates high, modelling how real classifiers behave rather than being chosen to flatter — most mass lands in the top bin, which is why equal-width binning leaves the middle of the range nearly empty.

`bedrock` and `vertex` defer their imports into the constructor so `import ai_classifier` works without the extras. Their `parse_response` is module-level and takes a plain dict, so the interesting failures are testable without a network. It **raises** on anything unreadable rather than defaulting to `not_enough_info`, because defaulting would convert a broken integration into a pile of plausible abstentions.

### core/batch.py — checkpoint and resume

Append-only JSON Lines, flushed per record. A rewritten snapshot has a window during which the file is partial, and a crash inside it loses everything already done — the exact scenario a checkpoint exists for. Appending means the worst case is one truncated final line, which is detectable and reported.

One provider per checkpoint, enforced. Resuming under a different provider would blend two providers' verdicts into one calibration measurement, describing neither.

Failures are **not** checkpointed: a transient error should be retried, and recording it would make it permanent.

### render/ — three surfaces, one rule

`n/a` and `null`, never `0`. `text.py` is primary because it asserts in CI; `chart.py` is nicer for a human and can only be verified by looking at it.

---

## The Bugs This Codebase Had

### A metric with an empty denominator reported a perfect score

ECE is a distance, so `0.0` is the best attainable value — and it was exactly what a subgroup with no answers returned. Ranking modalities by ECE:

```
ECE=0.0000  image_only  (answered   0 of  43)   <- best calibrated?
ECE=0.0000  neither     (answered   0 of   7)   <- best calibrated?
ECE=0.0802  text_only   (answered  80 of  82)
ECE=0.1574  both        (answered 143 of 148)
```

The two subgroups the provider is blind to ranked as its best. An abstain-on-everything provider scored a perfect headline. Simultaneously `accuracy` returned `0.0` on those same subgroups, reading as total failure — so two headline metrics contradicted each other, both artefacts of the same empty denominator.

Fixed by making both `float | None`, which turns the offending comparison into a `TypeError`, plus `blind_spots()` to name what was not measured. Full account in **ADR-002**.

### A negation fired the very term it negated

A test failed and I assumed my premise was wrong. It was not: `"flammable"` is a **substring** of `"non flammable"`. Scanning the supporting and contradicting tables independently made a listing reading "non flammable" score 3 supporting *and* 3 contradicting — exact cancellation — so the provider abstained on unambiguous evidence.

The presentation is what made it survive: **an abstention reads as caution.** No exception, no implausible verdict, just a slightly elevated abstention rate. And it compounded the bug above, since more abstentions meant more blind spots in a metric that rewarded them.

Fixed by scanning all terms longest-first and blanking each matched span. Word boundaries would not have helped — `flammable` sits on clean boundaries inside `non flammable`. `setup.sh` derives the collision set from the tables rather than hard-coding it.

### Two smaller ones, caught in self-review

**A sentinel violated its own field constraint.** `choose_threshold` encoded "unreachable" as `value=1.01`, on a field declared `Field(ge=0.0, le=1.0)`. The one branch that most needed to be reportable raised a validation error. Replaced with `None`.

**The modality breakdown was silently rebinned.** `render` passed `bin_count=len(report.bins)`, but `bins` holds only *populated* bins — 5 of 10 requested on this data. So the table under the headline was computed at half the resolution, and nothing about the output would have looked wrong. Fixed by carrying `bin_count` on the report.

---

## Data Flow

```
1.  ai-classifier measure -i data/products.csv -l data/labels.csv
2.  load_products -> 340 products
      blank attributes dropped; images split on , or |
      modality derived: 181 both, 99 text_only, 53 image_only, 7 neither
3.  load_labels -> 280 usable, 19 not_enough_info collected as undecided
      (undecided are review items, NOT ground truth -- ADR-001)
4.  provider.classify per product
      "contains non flammable"
        -> longest-first: "non flammable" matches, span blanked
        -> "flammable" can no longer match inside it
        -> FALSE_POSITIVE, margin -3
      image_only listing
        -> supports(IMAGE_ONLY) is False
        -> unable(): NOT_ENOUGH_INFO, confidence 0.0, reason recorded
5.  pair() on product_id
      280 outcomes, 60 unlabelled (reported), 0 unclassified
      duplicates in either input would RAISE
6.  calibrate
      223 answered, 57 abstained
      abstentions excluded from bins; 4 populated bins of 10 requested
      accuracy 83.9%, ECE 0.1297, overconfident
7.  calibrate_by_modality
      image_only: 43 items, 0 answered -> ece None, accuracy None
      neither:     7 items, 0 answered -> ece None, accuracy None
8.  choose_threshold(target=0.95)
      no cut-off reaches it; peak is 83.9% -> value None, automate nothing
9.  render: n/a in the table, blind spots named in the body
10. exit 1 -- twice over: a blind spot AND an unreachable target
```

Observed by `setup.sh`:

```
     340 products, 280 usable labels, 19 undecided (excluded)
     280 scored, 60 unlabelled
     accuracy 83.9%, ece 0.1297, overconfident
     image_only: 43 items, 0 answered, ece not measurable
     blind spots reported: ['image_only', 'neither']
     order-independent across 8 shuffles
     100 reused, 240 newly run, 340 total
     a truncated line was reported (line 341) and the run still completed
     switching provider mid-checkpoint is refused
```

---

## Configuration

`AC_`-prefixed environment variables or a `.env` file. No list-typed fields, since Pydantic Settings parses those as JSON and `AC_PROVIDERS=offline,vertex` would fail with a decode error that reads like a bug in the tool.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AC_", env_file=".env", extra="ignore")

    products: Path = Path("data/products.csv")
    labels: Path = Path("data/labels.csv")
    output_dir: Path = Path("output")
    provider: str = "offline"
    bin_count: int = 10
    target_accuracy: float = 0.95
```

---

## Testing

```bash
pytest                          # 130 tests, ~0.3s
pytest --cov=ai_classifier --cov-report=term-missing
```

| File | Covers |
|:--|:--|
| `conftest.py` | the shipped catalogue as session fixtures; `many()` builds a population whose true accuracy is known exactly, so a calibration gap is asserted against arithmetic rather than a previous run |
| `test_calibrate.py` | binning incl. the 1.0 boundary; abstention exclusion; **the empty-denominator regression class**; blind spots; thresholds; feedback and stability; property tests over the catalogue |
| `test_models_providers_io.py` | the three refusals (confident abstention, undecided label, mismatched outcome); modality detection; **the substring collision**; provider registry; Bedrock response parsing incl. five malformed payloads; loading; pairing; checkpoint resume |
| `test_render_export.py` | `n/a` on four surfaces; the diagram; threshold advice; null-not-zero in JSON and CSV; PNG magic bytes |
| `integration/test_cli.py` | all five commands, both exit-code reasons, stdout staying pipeable |

The property tests run against `data/`, so editing it and breaking an invariant fails the build. Two of them exist to stop the others being vacuous: every modality and every verdict must be exercised by the catalogue.

Not covered: the two cloud `classify` methods, which need credentials, and the matplotlib-missing branch — all marked `pragma: no cover`. Their response parsing is fully covered, which is where the failures live.
