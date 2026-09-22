# Phase 3 + 4 — the hypothesis does not survive an honest split

Measured 2026-09-21 on `data/labeled/master.csv` (37,344 rows after excluding
the 30 contradictory ones). Scripts: `src/models/phase3_alarms.py`,
`src/models/phase4_generalisation.py`. Raw logs:
`docs/evidence/phase3_alarms.txt`, `docs/evidence/phase4_generalisation.txt`.

## The headline

```
split scheme          balanced_acc    macro_F1   verdict
random (optimistic)       81.0%         81.8%    looks great
unseen-producer           54.0%         50.3%    near chance
unseen-lab                68.3%         67.6%    unstable (+/-18.6)
dual-class + unseen       54.6%         50.7%    near chance
```

Balanced-accuracy chance level is **50.0%**.

A random split says the chemistry separates solventless from hydrocarbon at
81%. Hold out the **producer** and it collapses to 54% — four points above
coin-flipping. The 27-point gap is the measurement of how much that 81% was
producer identity rather than extraction chemistry.

**On this data, the hypothesis fails.** The cannabinoid/terpene profile in
public COAs does not carry enough information to identify extraction method
for a producer the model has never seen.

## Phase 3 raised the alarm first

Before touching the real task, three probes asked whether the chemistry
predicts things it should not:

```
target      classes   rows    accuracy   baseline    lift
lab              11  37276      55.6%      40.3%   +15.3 pts  [ALARM]
producer         33  30416      24.3%      14.4%    +9.9 pts  [watch]
```

The profile **does** fingerprint the lab. That is not surprising in hindsight
— instruments, calibration and reporting conventions differ — but it means any
score achieved while train and test share labs is partly measuring the
instrument.

This is exactly why Phase 4's splits were built before trusting any number.

## Why the strictest test matters most

`dual-class + unseen` is the strongest evidence available here. It trains and
tests only on the **53 producers that make both** solventless and hydrocarbon
extracts, grouped so no producer spans train and test.

Inside that subset, knowing the maker tells you nothing about the label — so
any lift has to come from the chemistry. The result is **54.6%**, against a
50.0% chance level.

That is the honest number for the original question.

## What the confusion matrices show

```
unseen-producer     true hydrocarbon: 16149 correct / 5818 wrong
                    true solventless:  3332 correct / 7930 wrong
```

Solventless recall is **32.3%**. The model mostly answers "hydrocarbon" — the
majority class — and its apparent skill is concentrated on the easy side. A
single balanced-accuracy figure would have hidden this; recall per class does
not.

`unseen-lab` at ±18.6 standard deviation across 8 folds is equally telling:
performance swings wildly depending on which lab is held out, which is the
signature of a model keyed to lab conventions rather than to a stable
chemical difference.

## What this does NOT prove

- **Not** that extraction method leaves no chemical trace. Controlled studies
  (MassIVE, the Zenodo Ecuador set with its explicit `Metodo` column) measure
  method effects under fixed genetics and fixed instrumentation. This says
  nothing about those.
- **Not** that a better model would fail. It says the *features available in
  public COAs* — 19 analytes, heavily right-censored — do not generalise
  across producers. A richer assay (full terpene panel, LC-MS fingerprints)
  is a different experiment.
- **Not** that the labels are wrong. They come from the declared
  `product_type`, cross-validated at 99.3% against product names.

The honest statement is narrow and specific: **strain and producer variation
dominate whatever method signal exists in these 19 analytes.**

## The confounder that remains untested

Strain is the obvious candidate. Solventless production skews toward cultivars
that wash well, so "solventless vs hydrocarbon" may partly encode "which
cultivar" — and cultivar drives terpene profile directly. `strain_name` is in
the master dataset and has not been probed yet.

That is the next measurement worth making, and it is a narrowing question, not
a rescue attempt.

## Roadmap status

Gate 4 asked:

> Does macro F1 on the unseen-brand test beat the majority class by more than
> the between-fold spread?

**No.** Macro F1 is 50.3% unseen-producer and 50.7% on dual-class producers,
against a 50.0% chance level, with fold spreads of 5.6 and 2.6 points.

Per the roadmap, that is a publishable negative result, not a failure — and it
was reached without ever fitting the answer the hypothesis wanted.
