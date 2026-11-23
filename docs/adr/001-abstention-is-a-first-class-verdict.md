# ADR 001: Abstention is a first-class verdict, and every metric must decide what to do with it

## Status

Accepted

## Context

A classifier that can say "I don't know" is a fundamentally different object from one that cannot, and the difference propagates into every number computed downstream.

The spec's hard cases make abstention unavoidable. An **image-only listing** carries pictures and no usable text, so a text provider has nothing to read. An **ambiguous product** — is it a toy or a collectible? — has evidence pointing both ways. A **contradictory listing** says "lithium battery" and "battery free" in the same description. In all three the honest answer is that the evidence does not settle it.

Forcing a binary verdict on those cases produces a guess with a confidence that cannot be anything but arbitrary. But admitting a third value means every aggregate has to answer a question it would rather not: does an abstention count as a wrong answer?

Both obvious choices are wrong.

**Count abstentions as errors.** Then a classifier that declines a case it genuinely cannot see scores worse than one that guesses and happens to be right half the time. That is backwards for a review pipeline, where "route this to a human" is the correct handling and a confident wrong answer is the expensive failure.

**Count abstentions as successes.** Absurd on its face, but it is what happens implicitly if abstentions are filtered out of the denominator *and* nothing reports how many were filtered. Accuracy climbs as the classifier declines more, and the report shows no trace.

## Decision

`NOT_ENOUGH_INFO` sits in the same enum as the two answers, with a property that names what it is not:

```python
class Verdict(StrEnum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    NOT_ENOUGH_INFO = "not_enough_info"

    @property
    def is_answer(self) -> bool:
        return self is not Verdict.NOT_ENOUGH_INFO
```

In the enum rather than modelled as `verdict=None`, because "I looked and could not tell" is a real conclusion a reviewer needs to see, and because a nullable field invites `if verdict:` which silently treats it as absent data.

### The consequences, each decided explicitly

**`Outcome.correct` returns `True`, `False`, or `None`.** Not a boolean. Three things can happen and they need different responses. A caller writing `if outcome.correct:` treats an abstention as wrong; `if outcome.correct is False:` does not. The three-valued return does not prevent the mistake, but it makes the wrong reading visibly a choice rather than an invisible default.

**Accuracy is computed over answers only, and never travels alone.** Excluding abstentions is right — declining is not a mistake — and it is gameable, so `coverage` and `abstention_rate` sit beside it in every report and every export. The pair cannot be gamed, only traded off: a classifier can have high accuracy or high coverage, and stating both makes the trade visible.

**Abstentions are excluded from reliability bins.** This is the load-bearing one. There is no answer whose correctness could be compared against a confidence, so an abstention has nothing to contribute to a reliability curve. Including them would be actively perverse: abstentions carry low confidence by construction, so they would land in the low-confidence bins, and **low accuracy in a low-confidence bin reads as good calibration**. A provider that abstains heavily would score as better calibrated than one that commits.

There is a test asserting that adding 200 abstentions leaves the ECE unchanged to four decimal places.

**Abstentions ARE included in the confusion matrix.** They are excluded from calibration but the *shape* of what a classifier declines is diagnostic. Abstaining mostly on items whose true label is `FALSE_POSITIVE` is a different problem from abstaining at random, and only the matrix shows that.

**A confident abstention is refused at construction.**

```python
if self.verdict is Verdict.NOT_ENOUGH_INFO and self.confidence > 0.5:
    raise ValueError(...)
```

"I cannot tell, and I am 97% sure of my answer" is incoherent, but an LLM asked for a verdict and a confidence will happily emit it — models put 0.95 on everything. Left alone it poisons calibration from both ends: it inflates the high-confidence bins with items that were never answers, and it hides the abstention from any reviewer filtering on low confidence.

Refused rather than repaired. The sensible-looking repair — rewrite the confidence to something low — invents a number the provider never reported.

**A LABEL of `not_enough_info` is refused outright.** A reviewer who could not decide has not produced a target to measure against. Admitting these would make "the classifier abstained and the human also abstained" score as a correct answer, which would let a provider that abstains on everything look accurate. Such rows are legitimate data — they mark products needing better evidence — so `load_labels` collects them into `undecided_labels` and reports the count rather than dropping them silently.

**A provider that cannot read a product still returns a Classification.** `Provider.unable()` produces an honest abstention with confidence 0.0 and a stated reason. Filtering unsupported products out instead would remove them from the output entirely, so a reviewer would never learn they needed attention — the worst outcome for exactly the listings the tool exists to surface.

## Validation

- `TestVerdict`, `TestOutcome` — `is_answer`, and `correct` being three-valued
- `TestClassificationRefusesIncoherentOutput` — the confident abstention, at three confidences
- `TestLabelMustBeGroundTruth` — an undecided label raises
- `TestAbstentionsAreExcludedFromCalibration` — bins hold only answers; 200 added abstentions do not move the ECE; accuracy is over answers only
- `TestConfusionIncludesAbstentions` — they appear in the matrix
- `TestOfflineProvider` — abstains on image-only, on no-signal, and on an exact tie, each with a distinct reason
- `setup.sh` re-derives the ECE-invariance and the confident-abstention refusal on a fresh clone

## Consequences

**Good**

- The three hard cases from the spec have an honest answer instead of a forced guess
- Abstaining cannot improve a calibration score, so the metric cannot be gamed by refusing to answer
- Abstaining cannot be punished as an error either, so an honest provider is not penalised against a guesser
- A reviewer sees *why* something was declined, because the reason travels on the classification
- The undecided-label case is visible rather than a silently shrinking denominator

**Bad**

- Three-valued logic in the place readers least expect it. `correct: bool | None` is correct and it is a trap for anyone who skims the type, and the guard is a docstring rather than anything enforced
- Every aggregate function had to decide about abstentions separately. There is no single place that handles it, so a new metric added later can get it wrong independently
- Accuracy excluding abstentions means the headline number rises as coverage falls. `coverage` sits next to it, but a reader skimming one column still gets a flattering impression
- The `> 0.5` cut-off for a confident abstention is arbitrary. A provider reporting 0.5 exactly passes, and there is no principled reason 0.5 is the line rather than 0.3

**Neutral**

- `NOT_ENOUGH_INFO` in the same enum as the answers means every `match` over `Verdict` needs three arms. Making it a separate type would force the distinction at compile time and would triple the plumbing
- `Provider.unable()` is a base-class helper rather than each provider phrasing its own refusal, so all three say "I can't" identically. That is deliberate — a reviewer filtering on rationale should not need to know which backend ran — but it does mean the wording is shared state
- Confidence 0.0 on an *answer* is legal, though nothing produces it. It would bin at the bottom and be measured normally, which is the right behaviour if a provider ever reports it
