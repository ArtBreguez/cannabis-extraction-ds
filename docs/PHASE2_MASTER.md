# Phase 2 — units, label agreement, and the master dataset

Measured 2026-09-21 on the intact Cannlytics file. Scripts:
`src/preprocessing/measure_units_and_labels.py`,
`src/preprocessing/build_master.py`. Raw log:
`docs/evidence/units_and_label_agreement.txt`.

## 1. Units: one scale, no conversion needed

`total_thc` median per lab, over structured-labeled concentrates:

```
lab                                    n    median      p05      p95  unit
g3 labs llc                        14367     86.35    70.47    97.36  percent
nv cann labs llc                   12083     82.27    58.58    91.90  percent
db labs llc                         2135     83.60    42.79    94.62  percent
certified ag lab llc                1806     76.59    40.81    87.47  percent
purevita labs, llc                   662     68.93    29.60    82.93  percent
... (12 labs total)

distinct units in use: ['percent']
```

**All 12 labs report percent.** A concentrate is 40–99% THC by weight, so
percent lands near 70 and mg/g would land near 700 — a factor of 10, easily
separated by the median. Nothing sits in the mg/g range.

`total_terpenes` corroborates: medians 0.20–4.38, inside the expected 0.5–15%
band for concentrates.

**Decision: no unit conversion is applied.** Adding one would introduce a
silent 10x bug with nothing to fix.

## 2. Label agreement: the regex path is trustworthy

Rows carrying **both** a structured label and a name-derived one:

```
rows with BOTH labels: 4284
  agree:    4254 (99.3%)
  disagree:   30 (0.7%)
```

Per-rule accuracy against the structured label:

```
rule             agree  disagree  accuracy
live_resin        1561        12     99.2%
shatter           1005         6     99.4%
badder             656         9     98.6%
bho                564         0    100.0%
live_rosin         231         2     99.1%
rosin              144         1     99.3%
hash_rosin          83         0    100.0%
wax                 10         0    100.0%
```

99.3% agreement is strong enough that the name path is kept — but only to
**flag conflicts**, never to overwrite the declared label.

## 3. Two real bugs the tests caught

Writing tests for the rules exposed defects that a green suite would have
hidden.

**Texture is not method.** `badder`, `wax` and `shatter` describe
*consistency*, not extraction. 41 rows are named things like `shango alien
banana candy live rosin badder` — rosin (solventless) that happens to have
badder texture. A rule order putting texture first would mislabel every one of
them as hydrocarbon.

**`\bbho\b` misses `bho100223`.** Real product names glue a batch code onto
the token (`gelato 33 badder bho100223`). With a strict word boundary the
`bho` rule never fires and the row falls through to `badder` — hydrocarbon by
luck, not by evidence. Fixed to `\bbho(?![a-z])`, which still rejects
`bhopal`.

Both are covered by tests, and 6 mutants of the rule table were killed (rule
order inverted, empty-as-zero, strict `\bbho\b` restored, live_resin flipped
to solventless, texture confidence promoted to high, `wax` boundary removed).

## 4. The 30 conflicts are a source-data error

```
product_type EXACT on conflicting rows:
    66  non-solvent based concentrate
     2  non-solvent based concentrate (each)

spread over 5 labs and many producers
```

The structured field says **non-solvent based** for products named *live
resin*, *shatter*, *BHO*. That is physically contradictory: live resin is by
definition hydrocarbon-extracted from fresh-frozen material.

Somebody mislabeled at source, and there is no way to tell which side is
wrong. These rows are **kept and flagged** (`label_conflict=1`), excluded from
the default training view, and left visible for audit. They are 0.08% of the
labeled data.

## 5. Master dataset

`data/labeled/master.csv` — 11 MB, 37,374 rows, 82 columns.

```
solventless:      12,798
hydrocarbon:      24,576
label_conflict:        30  (flagged, excluded by default)
analyte columns:      34  (+34 _tested flags)
```

Every row carries provenance: `label_source`, `name_rule`, `name_label`,
`name_confidence`, plus producer, brand, lab, state, strain, batch, test date
and COA URL.

**Integrity verified after the build:**

```
value/flag incoherence:              0 rows
d_limonene measured-zero:        8,274
d_limonene not-tested:           7,749
```

The zero-vs-empty distinction survives the write. That was the whole point of
the `*_tested` design, and it is now checked rather than assumed.

## Status

Phase 2 is complete. Every decision the handoff left open is now answered by
measurement:

- non-detect encoding → value + `*_tested` mask; `alpha_terpinene` dropped
- unit normalisation → not needed, all labs report percent
- label cross-validation → 99.3% agreement, name path used only to flag

Phase 3 (confounder EDA) can start against `data/labeled/master.csv`.
The majority-class baseline to beat is **65.8%**.
