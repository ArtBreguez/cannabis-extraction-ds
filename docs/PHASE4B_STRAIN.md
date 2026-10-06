# Phase 4b — strain confounder: untestable with this data

Measured 2026-09-21. Script: `src/models/phase4b_strain.py`. Raw log:
`docs/evidence/phase4b_strain.txt`.

## Conclusion first

**The strain confounder cannot be tested with the Cannlytics data.** This is a
data-availability limit, not a finding. The numbers below are reported for
transparency but are not trustworthy, and none of them changes the Phase 4
conclusion.

## Why it cannot be tested

The dedicated cultivar columns are empty for concentrates:

```
strain_name filled in labeled rows: 0 of 37,374 (0.0%)
strain_type filled in labeled rows: 0 of 37,374 (0.0%)
```

Both columns exist in the source file but carry no data for concentrate
products. The only remaining cultivar signal is inside `product_name`, and
extracting a strain key from it produces mostly garbage:

```
top extracted "strain" keys:
  4717  (empty)
  2889  oil
  1543  thc
  1057  lab sample
   875  acres
   323  ethanol
```

22.6% of the extracted keys are clearly not cultivars (`oil`, `thc`,
`lab sample`, `ethanol`, `acres`, `raw`), and a further 4,717 rows yield no key
at all, so 32.4% of labelled rows carry no usable cultivar signal. Real strain
names (`blue dream`, `jack herer`, `wedding cake`) exist but are a minority, and
each appears in tens of rows at most. A confounder probe built on a key that
dirty cannot separate "the cultivar leaks" from "the extraction
artefact of my regex leaks".

## What the run produced anyway

```
chemistry -> strain key:   acc 19.5% vs baseline 22.5%  (-3.0 pts)
unseen-strain (all):       balanced_acc 66.2%  macro_F1 65.0%
dual-strain + unseen:      balanced_acc 56.9%  macro_F1 42.8%  (+/-11.4)
```

Read with heavy caution:

- The chemistry does **not** predict the (noisy) strain key — negative lift.
  If anything this hints strain is less of a fingerprint than lab was, but the
  key is too dirty to trust.
- `dual-strain + unseen` lands at 56.9% balanced accuracy with a macro F1 of
  **42.8%** — below the 50% chance level, with a ±11.4 fold spread. That is
  consistent with Phase 4's "near chance", but the instability makes it
  evidence of nothing on its own.

## What this means for the project

Phase 4 already answered the original question: **NO**, the 19 public-COA
analytes do not generalise across producers (54.6% balanced accuracy on
dual-class producers, chance is 50%). The strain probe was meant to explain
*why* — to confirm cultivar as the dominant confounder.

It cannot, because the data to identify cultivar is not in this dataset. The
honest status of the strain question is **open, and unanswerable with
Cannlytics alone**.

To actually test it would require a source that pairs concentrate chemistry
with a reliable cultivar label — which points back to the controlled datasets
noted in the audit (MassIVE, the Zenodo Ecuador set), not to more work on
Cannlytics.

## Net position of the research

- Original hypothesis: **not supported** on public COA data (Phase 4).
- Dominant-confounder identification: **producer**, confirmed; **strain**,
  plausible but untestable here.
- The negative result stands and is not weakened by the strain probe's
  failure — that probe simply could not run on data that does not exist.
