# Phase 2 — leakage and encoding audit

Measured 2026-09-21 on the intact Cannlytics file (2,534.8 MB), over the
37,374 rows carrying a structured extraction-method label.

Scripts: `src/preprocessing/measure_nondetects.py`,
`src/preprocessing/measure_lab_confounding.py`.
Raw logs: `docs/evidence/nondetect_encoding.txt`,
`docs/evidence/lab_confounding.txt`.

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

## 3. Leakage probe — the good news

A classifier built on **missingness alone** (which terpenes were reported,
ignoring every measured value):

```
accuracy from missingness alone: 65.8%
majority-class baseline:         65.8%
lift over baseline:              +0.0 points
```

**Zero.** The cheapest lab-shaped shortcut buys nothing. Adding lab identity
on top reaches only 67.8% — +2.0 points over always guessing the majority.

So the coverage gap in §2 is real but **not a usable shortcut**. Any model
that performs well will have to do it on measured chemistry.

This is a genuine result, not an assumption, and it is what makes Phase 4
worth running.

## 4. Labs are not entangled with class either

Only **373 rows (1.0%)** sit in labs that are ≥90% one class. The two largest
labs — `g3 labs llc` (15,035 rows) and `nv cann labs llc` (12,539) — are both
close to the overall 66/34 split.

One lab to watch: `green peaks analytical`, 100 rows, 100% solventless. Small,
but it should never be the only lab in a test fold.

## 5. Group-split feasibility

```
producers in both classes: 55 of 96
rows belonging to them:    28,927 (77.4% of labeled data)
class balance:             24,576 hydrocarbon / 12,798 solventless (65.8/34.2)
```

77.4% of the labeled data comes from producers that make **both** solventless
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
- primary evaluation on the 55 dual-class producers
- `GroupKFold` by producer, plus an unseen-lab split
- majority-class baseline is **65.8%** — accuracy below that is worthless, and
  accuracy near it means nothing was learned

## Still open

- Unit normalisation (percent vs mg/g) across labs — not yet measured.
- Cross-validating the name-based label against the structured one.
