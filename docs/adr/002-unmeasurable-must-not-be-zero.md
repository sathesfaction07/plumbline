# ADR 002: A metric with an empty denominator must report nothing, not zero

## Status

Accepted

## Context

Expected Calibration Error is the weighted mean absolute gap between claimed confidence and observed accuracy across bins. It is a **distance**, so lower is better and **zero is perfect**.

That makes the empty case dangerous in a way most metrics are not. Consider a subgroup the classifier never answered — every item an abstention. Abstentions are excluded from reliability bins (ADR-001), so the subgroup has no bins. The natural implementation:

```python
total = sum(item.count for item in bins)
if total == 0:
    return 0.0                       # looks defensive; is not
return sum(abs(item.gap) * item.count for item in bins) / total
```

`0.0` is the **best attainable score.** Returning it for "nothing was measured" does not merely lose information — it awards a perfect result for having answered nothing at all.

### What that produced

Ranking modalities by ECE on the shipped catalogue, best first:

```
ECE=0.0000  image_only  (answered   0 of  43)   <- best calibrated?
ECE=0.0000  neither     (answered   0 of   7)   <- best calibrated?
ECE=0.0802  text_only   (answered  80 of  82)
ECE=0.1574  both        (answered 143 of 148)
```

The two subgroups the provider is completely blind to came out as its best-calibrated segments. A dashboard sorting ascending puts them at the top. **This is an inversion, not an ambiguity** — the metric does not merely fail to say anything, it says the opposite of the truth. And image-only listings are precisely the case this project exists to handle.

Verified at the extreme: a provider abstaining on all 280 items scored a perfect headline ECE of `0.0000`.

### And the same defect ran the other way at the same time

On that same subgroup, `accuracy` returned `0.0`, because it was `correct / answered` guarded with `if answered == 0: return 0.0`. Zero accuracy reads as **total failure**.

So the two headline metrics pointed in opposite directions on identical data — one saying flawless, one saying catastrophic — and both were artefacts of the same empty denominator. Neither was a measurement.

The honest information was already present. `coverage` was `0.0%`, `answered` was `0`, `bins` was empty, `direction` was `"unknown"`. So this was never a missing measurement; it was a reporting defect, which is why it survived. Nothing crashed and every number looked plausible.

## Decision

**Make the undefined case unrepresentable as a number.**

```python
def expected_calibration_error(bins: list[Bin]) -> float | None:
    total = sum(item.count for item in bins)
    if total == 0:
        return None
    ...

class CalibrationReport(BaseModel):
    ece: float | None

    @property
    def measurable(self) -> bool:
        return self.answered > 0

    @property
    def accuracy(self) -> float | None:
        if self.answered == 0:
            return None
        return self.correct / self.answered
```

`None` rather than `NaN`, a flag, or a sentinel. The reason is what `None` makes impossible:

```python
sorted(reports.values(), key=lambda r: r.ece)
TypeError: '<' not supported between instances of 'NoneType' and 'float'
```

**The comparison that produced the inverted ranking now raises.** A flag alongside a float would not do that — a caller can ignore a flag. `NaN` is worse than either: it sorts unpredictably, compares False against everything including itself, and propagates silently through arithmetic. `None` fails loudly at the exact point where a blind spot would otherwise be ranked as a success.

### Naming the gap, because None is easy to skim past

Making the value `None` stops it being mistaken for a good score, but a table full of `n/a` still requires a reader to notice. The subgroup that most needs attention is the one with nothing in it. So `blind_spots()` turns the silence into a finding:

```
blind spots
  - image_only: 43 item(s), NONE answered. No calibration figure exists for this
    subgroup -- it was not measured, and a reported 0.0 would mean the opposite.
  - neither: 7 item(s), NONE answered. ...
```

Sorted by how much is unmeasured, largest first: 43 unanswered items matters more than 7. A thinly-answered subgroup gets a different message, because "only 3 of 23 answered" is a different problem from "none at all".

### Carried through every layer

The fix is worthless if a downstream layer re-introduces the zero.

- **Text render** — `n/a`, never `0.0000`, with a footnote saying that a `0.0` here would read as a perfect result
- **JSON export** — `null`, so `json.load(...)["ece"] < 0.05` cannot silently evaluate True on a subgroup nobody answered
- **CSV export** — empty cell, never `0`
- **Chart** — draws no curve and titles itself "not measured". A flat line would sit on the `y=0` axis, which reads as "always wrong" — a *third* distinct misreading of the same empty denominator
- **CLI** — `measure` exits **non-zero** when any subgroup is a blind spot, so a pipeline gates on it rather than printing a good-looking number

`doctor` explains the convention unprompted, because a reader encountering `n/a` deserves to know it is a deliberate refusal rather than a formatting gap.

## Validation

`TestEmptyDenominatorMustNotLookLikeAPerfectScore` is the regression class:

- `ece` is `None`, not `0.0`, when nothing was answered
- `accuracy` is `None`, not `0.0` — the opposite-direction half
- `expected_calibration_error([])` is `None`
- a provider abstaining on all 280 items gets no score
- **sorting a mixed set of reports by `ece` raises `TypeError`** — the strongest guarantee available
- `measurable` is the flag to check first

Plus `TestBlindSpots` (naming, ordering, the thin-subgroup variant), `TestUnmeasurableRendersAsNotAvailable` (four render surfaces), `TestExport` (null not zero, in JSON and CSV), and CLI tests asserting the exit code and that the `image_only` row contains `n/a` and not `0.0000`.

`setup.sh` and CI both re-derive it on a fresh clone, including the `TypeError` on sort.

## Consequences

**Good**

- The inverted ranking is impossible: the comparison raises rather than misleading
- Two metrics that used to contradict each other on the same data now both decline to answer
- A blind spot is a named finding with a count, not a blank cell
- `measure` gates a pipeline on "did we actually measure this", which is a better question than "is the number small"
- The convention is uniform across text, JSON, CSV, chart, and exit code, so no layer re-introduces the zero

**Bad**

- `float | None` on the two most-read fields in the codebase. Every consumer needs a branch, and formatting helpers (`_pct`, `_ece`) exist solely to absorb it
- The `TypeError` on sort is a good outcome reached by an ugly mechanism. It is Python's dynamic typing catching it at runtime, not a type the caller was forced to handle. A reviewer who never sorts never learns the constraint exists
- `blind_spots()` uses the same `MIN_SAMPLES_PER_BIN = 10` threshold as bin reliability. Reusing one constant for two different judgements is convenient and not obviously correct — the right sample size for "trust this bin" and for "trust this subgroup" need not be the same number
- `measure` exiting non-zero on a blind spot means the shipped example fails by default, since the offline provider is blind to image-only listings. That is honest and it does mean a newcomer's first run "fails"

**Neutral**

- `direction` returns the string `"unknown"` rather than `None`, because it is already a string-valued summary and `"unknown"` cannot be mistaken for a good score the way `0.0` can
- `abstention_rate` and `coverage` still return `0.0` for an empty set. Both are fractions of the *total*, which is known even when nothing was answered, so zero is a real value there
- A zero-length span of the confidence range with genuinely zero gap would also report `0.0` — a real perfect score. It is indistinguishable in the output from... nothing, now, because the unmeasurable case no longer produces a number
