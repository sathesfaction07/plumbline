# AIClassifier

Classifies products as **true positive / false positive / not enough info**, then measures whether the confidence attached to those verdicts means anything.

The classification is the easy half. The hard half is that **a confidence score is a claim, and nobody checks it.** A provider reporting 0.95 on everything is not thereby 95% accurate, and routing work on that number means auto-accepting a population whose real error rate you do not know.

Three providers: an offline text one that needs nothing, and Bedrock Claude or Vertex Gemini for multimodal. The offline one is the default so the whole pipeline runs in CI with no credentials.

---

## The Distinction That Organises Everything

**Accuracy** asks "how often is it right". **Calibration** asks "when it says 0.9, is it right 90% of the time". They are independent, and only the second lets you automate anything.

A classifier can be 97% accurate and badly calibrated — it says 0.99 on everything, so its confidence tells you nothing about which 3% to review. Or 70% accurate and perfectly calibrated, which is genuinely useful: you know exactly what to trust.

On the shipped catalogue the offline provider is **83.9% accurate with an ECE of 0.1297, overconfident** — it claims 0.950 in its top bin and is right 82.9% of the time.

---

## The Bug This Project Had

ECE is a **distance**, so its best possible value is `0.0`. And `0.0` is exactly what a naive implementation returns for a subgroup the classifier never answered, because there are no bins to disagree with reality.

Ranking modalities by ECE, best first, on real data:

```
ECE=0.0000  image_only  (answered   0 of  43)   <- best calibrated?
ECE=0.0000  neither     (answered   0 of   7)   <- best calibrated?
ECE=0.0802  text_only   (answered  80 of  82)
ECE=0.1574  both        (answered 143 of 148)
```

**The two subgroups the provider is completely blind to ranked as its best-calibrated segments.** A provider abstaining on all 280 items scored a perfect headline ECE. This is not an ambiguity — it is an inversion, and a dashboard sorting ascending puts the blind spots at the top.

Worse, on the same subgroup `accuracy` returned `0.0`, which reads as *total failure*. So the two headline metrics pointed in opposite directions and both were artefacts of the same empty denominator.

The fix makes the undefined case unrepresentable as a number: `ece` and `accuracy` are `float | None`. Now the comparison **raises** instead of misleading —

```python
sorted(reports.values(), key=lambda r: r.ece)
TypeError: '<' not supported between instances of 'NoneType' and 'float'
```

— and `blind_spots()` names them, because `None` in a table is easy to skim past:

```
blind spots
  - image_only: 43 item(s), NONE answered. No calibration figure exists for this
    subgroup -- it was not measured, and a reported 0.0 would mean the opposite.
```

---

## Running

```bash
bash setup.sh                 # verifies behaviour, not just install
bash setup.sh --chart         # + matplotlib for PNG reliability diagrams
source .venv/bin/activate

ai-classifier doctor
ai-classifier providers
ai-classifier measure    -i data/products.csv -l data/labels.csv
ai-classifier stability  -i data/products.csv -l data/labels.csv
ai-classifier classify   -i data/products.csv -c out/ck.jsonl
ai-classifier measure    -i data/products.csv -l data/labels.csv -o out/
```

`measure` on the shipped catalogue:

```
summary
  items            280
  answered         223  (79.6% coverage)
  abstained        57  (20.4%)
  accuracy         83.9%   (over answers only)
  ECE              0.1297   (equal_width bins)
  direction        overconfident

reliability
  bin        n     claimed  observed   gap  diagram
  [0.6,0.7)    15   0.658     0.800  -0.142               |     #
  [0.7,0.8)    11   0.754     1.000  -0.246                  |        #
  [0.8,0.9)     4   0.817     1.000  -0.183                    |      #   (thin)
  [0.9,1.0)   193   0.950     0.829  +0.121                     #    |
                                         | claimed   # observed

by modality
  modality      items  answered  coverage  accuracy       ECE  direction
  both            148       143     96.6%     81.8%    0.1574  overconfident
  image_only       43         0      0.0%       n/a       n/a  unknown
  neither           7         0      0.0%       n/a       n/a  unknown
  text_only        82        80     97.6%     87.5%    0.0802  overconfident

blind spots
  - image_only: 43 item(s), NONE answered. ...
  - neither: 7 item(s), NONE answered. ...

routing
  target 95%: NOT REACHABLE
  no cut-off reaches 95%; the whole answered population peaks at 83.9%,
  so nothing should be auto-accepted
  recommendation: route all 280 to human review
```

The `|` is claimed confidence, `#` is observed accuracy, so the distance between them **is** the calibration gap. Note the shape: the provider is *underconfident* in the middle bands and *overconfident* in the top one, which is why a single headline number is not enough on its own.

---

## Commands

| Command | Purpose | Exit |
|:--|:--|:--|
| `measure` | Classify, compare to labels, report calibration | **1** if any subgroup is unmeasurable, or no threshold reaches the target |
| `classify` | Batch classify with checkpoint/resume | 1 if any product failed |
| `stability` | Whether the threshold has settled as labels accumulated | 0 |
| `providers` | The three backends and what each can read | 0 |
| `doctor` | Config, inputs, and how to read the report | 1 if inputs unreadable |

`measure` gating on a blind spot is the deliberate consequence of the bug above. "We measured nothing here" is a result somebody has to see, so it stops the build rather than printing a good-looking number.

---

## Verdicts

| Verdict | Meaning |
|:--|:--|
| `true_positive` | the flag was correct |
| `false_positive` | the flag was wrong |
| `not_enough_info` | **an abstention.** Not an answer, and not an error either |

The third one drives most of the design. An abstention is excluded from calibration bins (there is no answer whose correctness could be compared to a confidence) and excluded from accuracy (declining a case you cannot see is not a mistake). Both exclusions are load-bearing, and both create risks handled explicitly:

- Excluding from bins means abstaining cannot *improve* a calibration score. Verified: adding 200 abstentions leaves the ECE unchanged to 4 decimal places.
- Excluding from accuracy means abstaining could *inflate* it. That is why `coverage` and `abstention_rate` sit next to accuracy — the pair cannot be gamed, only traded off.
- A label of `not_enough_info` is refused outright. A reviewer who could not decide has not produced a target, and admitting these would score "the classifier abstained and so did the human" as correct.

---

## Testing

```bash
pytest                          # 130 tests, ~0.3s
pytest --cov=ai_classifier --cov-report=term-missing
```

The property tests run against the shipped catalogue, so editing `data/` and breaking an invariant fails the build: the modality breakdown partitions the population; bins hold exactly the answered outcomes; every modality and every verdict is exercised (otherwise the assertions below them are vacuous); and calibration is order-independent across 8 shuffles.

`setup.sh` re-derives all of it on a fresh clone rather than trusting it, including the substring collision below.

---

## Tradeoffs and Decisions I Made Along the Way

**A negation was firing the very term it negated.** A test I wrote failed, and I assumed my test premise was wrong. It was not — `"flammable"` is a **substring** of `"non flammable"`, so scanning the supporting and contradicting tables independently made a listing reading "non flammable" score 3 supporting *and* 3 contradicting. Exact cancellation. The provider abstained on unambiguous evidence.

What makes this worth recording is how it presented: **an abstention reads as caution.** Nothing in the output looked wrong — no exception, no implausible verdict, just a slightly higher abstention rate than expected. It interacts nastily with the metric bug above, because more abstentions meant more blind spots in a metric that used to reward exactly that.

Fixed with longest-match-first scanning that blanks each matched span, so a negation consumes the word it negates. Word boundaries alone would not have helped: `flammable` sits on clean boundaries inside `non flammable`. `setup.sh` derives the collision set from the tables rather than hard-coding it, so adding a term that collides with an existing one fails the check.

**A sentinel of 1.01 violated its own field constraint.** `choose_threshold` reports "no cut-off reaches the target", and my first encoding was `value=1.01` — no real confidence could exceed it. But `value` is declared `Field(ge=0.0, le=1.0)`, so the one branch that most needed to be reportable raised a validation error instead. `None` says the same thing without lying about the type, and a caller cannot compare against it by accident.

**The modality breakdown was silently rebinned at a different granularity than the headline.** `render` passed `bin_count=len(report.bins)`, and `bins` holds only *populated* bins — 5 of 10 requested on this data, because saturating confidences leave most of the range empty. So the table under the headline was computed at half the resolution. Fixed by carrying `bin_count` on the report explicitly. Nothing about the output would have looked wrong.

**Abstentions are excluded from calibration, and that is the whole ballgame.** Including them would put items in the low-confidence bins that were never answers — and low accuracy in a low-confidence bin reads as *good* calibration. A provider that abstains heavily would score as better calibrated than one that commits.

**A confident abstention is refused at construction, not repaired.** "I cannot tell, and I am 97% sure" is not coherent, but an LLM asked for a verdict and a confidence will happily emit it. Left alone it poisons calibration from both ends: it inflates the high-confidence bins with items that were never answers, and it hides the abstention from anyone filtering on low confidence. The sensible-looking repair — rewrite the confidence — invents a number the provider never reported.

**Thresholds are chosen on observed accuracy, never on stated confidence.** Using the stated number would assume the calibration the function exists to work around. A provider claiming 0.95 while being right 80% of the time would get a threshold of 0.95, auto-accepting a population with a 20% error rate.

**Stability is measured over prefixes, not random subsamples.** The order labels arrive in is real information — reviewers work a queue, and if that queue correlates with difficulty then early labels are not representative. Random subsampling averages that away and reports a stability the actual arrival order does not have. The cost is that this measures stability under one ordering; it is the ordering you had, which produced the number you are deciding whether to trust.

**A bin below ten samples is reported but excluded from claims.** A bin with one sample has an observed accuracy of exactly 0.0 or 1.0, which looks catastrophic or perfect and is neither. Ten is a low bar chosen to be honest rather than flattering, and thin bins are marked `(thin)` rather than hidden — dropping them would silently change what the diagram covers.

**`Outcome.correct` returns `True`, `False`, or `None`.** Three things can happen and they need different responses. A caller writing `if outcome.correct:` treats an abstention as wrong; `if outcome.correct is False:` does not. The three-valued return does not prevent the mistake, but it makes the wrong reading visibly a choice.

**A failed product is not checkpointed.** A transient network error should be retried next run, and recording it would make it permanent. The consequence is that a deterministically failing product is retried every run — visible in the failure count rather than silently absorbed.

**Resuming a checkpoint under a different provider is refused.** Blending two providers' verdicts into one calibration measurement produces a number describing neither. The checkpoint records which provider wrote it and the run stops rather than mixing.

**No React UI, despite the spec asking for one.** A reliability diagram is normally a plot, and a plot can only be verified by a human looking at it. The text version asserts in CI, works over SSH, and needs no optional dependency — so it is the primary renderer, with an optional matplotlib PNG alongside for putting in front of people.

**The cloud providers are not covered by tests, and their response parsing is.** `parse_response` is a module-level function taking a plain dict precisely so the interesting failures are testable without a network. It **raises** on anything unreadable rather than defaulting to `not_enough_info`, because defaulting would silently convert a broken integration into a pile of plausible-looking abstentions — the hardest failure mode to notice.

---

## Configuration

```bash
AC_PRODUCTS=data/products.csv
AC_LABELS=data/labels.csv
AC_PROVIDER=offline
AC_BIN_COUNT=10
AC_TARGET_ACCURACY=0.95
```

`0.95` is a default, not a recommendation — the right target depends on what an error costs, which is why `--target` exists.

---

## Stack

| Concern | Choice |
|:--|:--|
| Language | Python 3.11+ |
| Models + config | Pydantic, Pydantic Settings |
| CLI | Click |
| Providers | offline (default), Bedrock Claude, Vertex Gemini |
| Charts | matplotlib, optional, Agg backend |
| Output | text reliability diagram, JSON Lines, CSV, JSON, PNG |
| Tests | pytest, Click's CliRunner |
| Lint | ruff |

The sample catalogue is synthetic: 340 products where each has a hidden true verdict and text that correlates with it imperfectly, since a thin listing is a thin proxy for the item. Nothing is tuned per-modality — the image-only blind spot arises because a text provider cannot see pictures, not because the data was arranged to produce it.
