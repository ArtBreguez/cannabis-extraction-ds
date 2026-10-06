# Phase 2 — leakage and encoding audit

Measured 2026-09-21 on the intact Cannlytics file (2,534.8 MB), over the
37,374 rows carrying a structured extraction-method label.

Scripts: `src/preprocessing/measure_nondetects.py`,
`src/preprocessing/measure_lab_confounding.py`.
Raw logs: `docs/evidence/nondetect_encoding.txt`,
`docs/evidence/lab_confounding.txt`.

Sections 3 to 5 were regenerated on 2026-10-06 after `measure_lab_confounding.py`
was changed to exclude the 30 rows whose `product_name` contradicts
`product_type`, so that their denominator (37,344) matches every model run.
Sections 1 and 2 still describe the 37,374-row labelled file.

## 1. Non-detects are already numeric — with one trap

Decision from `HANDOFF.md`, now answerable with data:

```
analyte                 numeric    empty    zero     ND   <LOQ
total_thc                 33597     3760      17      0      0
d_limonene                21351     7749    8274      0      0
beta_caryophyllene        22817     7744    6813      0      0
alpha_terpinene               0    37374       0      0      0
```

**No `ND` or `<LOQ` strings survive in this file** — Cannlytics has already
normalised them. So the four-way distinction the handoff worried about
collapses into a two-way one:

- **`zero`** — a real measurement that came back at/below detection.
- **`empty`** — not tested, or not reported. Absence of information.

These must not be merged. A zero is evidence; an empty cell is silence.

**`alpha_terpinene` is empty in 100% of labeled rows.** A column that is never
populated must be dropped, not imputed — imputing it would fabricate a feature
out of nothing.

### Consequence for the feature matrix

Every analyte gets two columns: the value, and a `*_tested` boolean. Rows are
never dropped for having gaps, and empty is never silently read as zero.

## 2. The real threat: coverage differs by class

```
analyte                 solventless  hydrocarbon
d_limonene                    40.8%        65.6%
beta_caryophyllene            47.1%        68.3%
alpha_pinene                  34.3%        56.3%
total_terpenes                56.9%        74.5%
```

Hydrocarbon rows carry terpene numbers far more often than solventless rows.
If that gap tracked **which lab** ran the test rather than the extract itself,
a classifier could infer the lab and score well while learning nothing about
extraction method — and Gate 4's unseen-brand test would collapse in the real
world.

So it was measured directly rather than assumed.

## 3. Leakage probe — a real but bounded shortcut

A classifier built on **missingness alone** (which of the 19 modelled analytes
carry a number, ignoring every measured value):

```
accuracy from missingness alone: 69.9%
majority-class baseline:         65.8%
lift over baseline:              +4.1 points
```

**+4.1 points.** An earlier version of this probe looked at 6 terpene columns
while the model saw 19, and reported +0.0. That was a lower bound on the wrong
quantity: a probe must see exactly the feature set the model sees. Adding lab
identity on top reaches 70.5%, a further 0.6 points.

The lift is not evenly spread. 7 of the 13 distinct missingness patterns are
**100% one class**, covering 4,123 rows (11.04%):

```
patterns that are 100% one class: 7 of 13
rows they cover:                  4123 (11.04%)
  n= 1495  all solventless (6 of 19 analytes reported)
  n= 1303  all hydrocarbon (1 of 19 analytes reported)
  n= 1303  all hydrocarbon (4 of 19 analytes reported)
```

For those rows the label is recoverable with certainty from which cells are
blank. They come from four labs (`PureVita`, `Cannalytics RI`,
`Lifted Testing`, `Green Peaks`) that run a restricted panel and record no
producer, so the shortcut is a reporting convention, not chemistry. Those are
also exactly the rows the unseen-producer scheme must drop, which is why the
same-sample comparison in PREPRINT §3.2 matters.

So the coverage gap in §2 is real and buys 4.1 points, concentrated in 11% of
the corpus. That is too small to explain a score in the eighties, which is
what makes Phase 4 worth running, but it is not zero and the manuscript now
says so.

## 4. Labs are not entangled with class either

Only **373 rows (1.0%)** sit in labs that are ≥90% one class. The two largest
labs — `g3 labs llc` (15,018 rows) and `nv cann labs llc` (12,531) — are both
close to the overall 66/34 split.

One lab to watch: `green peaks analytical`, 100 rows, 100% solventless. Small,
but it should never be the only lab in a test fold.

## 5. Group-split feasibility

```
producers in both classes: 53 of 96
rows belonging to them:    27,751 (74.3% of labeled data)
class balance:             24,573 hydrocarbon / 12,771 solventless (65.8/34.2)
```

74.3% of the labeled data comes from producers that make **both** solventless
and hydrocarbon extracts. Inside that subset, memorising the producer cannot
separate the classes — it is the strongest available test of whether the model
reads process rather than brand, and it is large enough to train on alone.

## What this changes in the roadmap

Phase 3's confounder checks were scheduled *before* modelling because a lab
shortcut would have invalidated everything downstream. The two cheapest
shortcuts are now measured and both are near-zero, so Phase 4 can proceed —
with these constraints:

- value + `*_tested` mask per analyte; empty is never zero
- drop `alpha_terpinene` (never populated)
- primary evaluation on the 53 dual-class producers
- `GroupKFold` by producer, plus an unseen-lab split
- majority-class baseline is **65.8%** — accuracy below that is worthless, and
  accuracy near it means nothing was learned

## Still open

- Unit normalisation (percent vs mg/g) across labs — not yet measured.
- Cross-validating the name-based label against the structured one.
