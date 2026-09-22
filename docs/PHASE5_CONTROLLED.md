# Phase 5 — controlled experiment (Zenodo 13823859, Ecuador)

Measured 2026-09-21. Script: `src/models/phase5_controlled.py`. Raw log:
`docs/evidence/phase5_controlled.txt`.

## What this dataset is

A designed experiment, not market data: 6 named varieties, each extracted by
3 lab methods (maceration, ultrasound, supercritical CO2) under varied
time/temperature/pressure, cannabinoids by HPLC. Fully balanced.

```
samples: 162   features: CBD, CBG, CBN, THC   (no terpenes)
methods:    Maceracion 54 / Supercritico 54 / Ultrasonido 54
varieties:  6 x 27, perfectly balanced
missing:    none
```

Two caveats that bound every claim below:

1. These are **lab methods**, not the market categories the project targets.
   Nothing here transfers to rosin-vs-BHO. It answers only the narrower
   question: *can extraction method leave a detectable chemical trace at all,
   with genetics controlled?*
2. 162 samples, 4 features. Small, so read the fold spreads.

## Result: the same pattern as Cannlytics

```
split                    balanced_acc   macro_F1
random 5-fold                56.7%        55.7%
leave-one-variety-out        38.9%        35.9%
```

Balanced-accuracy chance is **33.3%** (3 methods).

A random split shows method is somewhat separable (56.7%). Hold out an entire
variety and it drops to **38.9%** — 5.6 points above chance, with a ±12.6 fold
spread that straddles it. The same collapse Cannlytics showed when the
producer was held out.

## The alarm confirms why

```
chemistry -> variety:  balanced_acc 60.7%   (chance 16.7%)
```

The 4 cannabinoids predict the **variety** at 60.7% — nearly four times chance
— far better than they predict the extraction method. Even in a controlled
experiment built specifically to isolate method, genetics dominate the
cannabinoid signal.

## What this adds to the conclusion

The Cannlytics failure could have been blamed on messy market data. This
controlled experiment rules that out: even with fixed genetics, a designed
method comparison, and clean HPLC numbers, **method barely separates once you
require it to generalise across varieties, while variety itself is strongly
readable from the chemistry.**

Consistent story across two independent datasets:

- The chemical profile is dominated by **genetics** (variety/strain) and, in
  market data, by **producer and lab**.
- Extraction **method** is a weak signal that does not survive holding out the
  confounder — whether that confounder is producer (Cannlytics) or variety
  (this experiment).

## Honest limits of Phase 5

- 4 cannabinoids only. Terpenes are where method effects are most argued to
  show up (volatiles lost to heat/solvent), and they are absent here. A
  terpene-inclusive controlled set could still overturn this.
- 6 varieties is a small basis for "unseen variety"; the ±12.6 spread reflects
  that.
- Lab methods, not rosin/BHO.

## Where a real answer would come from

The remaining untested source is **MassIVE** (LC-MS/GC-MS method comparison),
which pairs method with a full metabolomic fingerprint including terpenes.
That is the only avenue left that could distinguish "method leaves no usable
trace" from "method leaves a trace, but not in the 4-cannabinoid / 19-analyte
projections tested so far". It is a mass-spec dataset (~GB), so it is a
separate build, not a quick extension.
