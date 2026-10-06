# Supporting Information: Group-aware validation sharply reduces the apparent accuracy of extraction-category classification in public cannabis laboratory data

**Arthur Gonçalves Breguez**
Independent researcher
ORCID: 0009-0005-8551-731X
Correspondence: arthurbreguez@gmail.com

This file holds the descriptive detail and the secondary tables that the main
text summarises: Sections S1 to S9 and Tables S1 to S8. Section numbers without
an S prefix, tables without an S prefix, Figure 1 and bracketed reference
numbers refer to the main text. Every figure given here is checked against the
same committed evidence logs as the main text, by
`src/evaluation/audit_preprint.py`.

## S1. From concentrate rows to usable rows

Most concentrate rows carry no declaration of the extraction family in
`product_type` (2.1) and are dropped at that step, which is where the bulk of
the reduction from concentrate rows to labelled rows happens:

**Table S1.** From concentrate rows to labelled rows.

```
concentrate or extract rows                     139,714
  product_type declares the extraction family    37,310   (26.7%)
  product_type does not                         102,404   (73.3%)

labelled rows outside that subset                    64
total labelled rows                              37,374
```

The discarded majority is dominated by values that name a format rather than a
method: `extracts` (34,034 rows), `concentrate, product inhalable` (30,586),
bare `concentrate` (26,030) and `marijuana extract for inhalation` (3,901).
These are not missing labels to be recovered, since the extraction family was
never recorded, so imputing one would invent the dependent variable. Matching is
by regular expression (`non[- ]?solvent`) rather than exact string, which also
admits regional variants such as `non-solvent concentrate` (1,445 rows) and, in
64 cases, `infused pre-rolls (non-solvent)`. Those 64 rows are the reason the
labelled total slightly exceeds the declared count within the concentrate
subset; they are solventless by declaration but are pre-rolls rather than
concentrates, they are 0.17% of the labelled data, and none names a producer, so
they are outside the headline comparison.

Of the 37,374 labelled rows, 30 carry contradictory labels (2.1) and are
excluded, which leaves the 37,344 usable rows of Tables S2 and S3.

**Table S2.** Class counts on the usable rows.

```
hydrocarbon   24,573   65.8%
solventless   12,771   34.2%
```

Grouping variables are populated as follows:

**Table S3.** Grouping variables on the usable rows.

```
producer      33,229 rows (89.0%)   96 distinct
lab           37,344 rows (100.0%)  12 distinct
strain_name        0 rows (0.0%)     0 distinct
```

## S2. Duplicate, split and degenerate records

Rows are not samples. 1,813 usable rows (4.9%) repeat another row on product
name, laboratory, producer and all 19 analyte values, but only 419 of them name
a producer; the other 1,394 are producer-less rows, which have no name and match
on a handful of analyte values. Separately, 1,303 samples from the producer-less
laboratories are stored as two adjacent rows each, with the same laboratory and
date and complementary analytes (`delta_9_thc` in one; `cbd`, `cbda`, `thca` and
`total_cbd` in the other), so those laboratories' 4,115 rows describe 2,812
samples. Among the rows that name a producer, 1,071 report nothing but a
`total_terpenes` of zero, in both classes, and two of the 96 producer strings
are spellings of one company (260 rows).

## S3. Product names that say ethanol, by year

Table S4 gives the counts behind the drift in the meaning of the label that 2.1
describes.

**Table S4.** Rows whose product name says ethanol, by year of test and declared
category. Dated rows only: 20 of the 163 non-solvent rows carry no date.

```
year    solvent-based   non-solvent
2020          82             9
2021          21             1
2022          98             0
2023          13           133
```

## S4. Analyte coverage and the total columns

The tested-versus-detected distinction of 2.2 produces two different and both
meaningful coverage figures, which we separate explicitly because conflating
them is easy:

**Table S5.** Share of the rows that name a producer with the analyte tested and
with a non-zero value, by class.

```
analyte               tested (flag)          detected (value > 0)
                   solventless  hydro      solventless  hydro
d_limonene              85.8%   90.7%           46.1%   73.4%
beta_caryophyllene      85.8%   90.8%           53.3%   76.4%
alpha_pinene            85.8%   90.8%           38.8%   63.0%
total_terpenes         100.0%  100.0%           64.4%   83.4%
```

Two properties of these columns are worth stating because they bound what the
features can mean.

First, `total_terpenes` is not an independent measurement. It is a near-exact
sum of the ten terpenes that survive the empty-column drop: on the 26,413 rows
that name a producer and report all ten, Pearson r = 0.992 against their sum and
R² = 0.984. So it is collinear with features already in the matrix and any
per-feature importance attributed to it is split credit rather than chemistry.
Its residual against that sum is laboratory-specific: `CERTIFIED AG LAB` reports
an exact sum (within 10⁻⁶) on 99.7% of rows, `MA & ASSOCIATES` on none, which
makes the residual a laboratory fingerprint in its own right. `total_thc` and
`total_cbd` are derived columns of the same kind, each a fixed combination of
its acid and neutral forms. Dropping the column costs 0.3 points of balanced
accuracy under the all-rows random split, so nothing in this paper depends on
keeping it; we keep it because the published literature does.

Second, whether `total_terpenes` is populated at all is a laboratory
convention, not a per-sample decision. Every laboratory is at 0% or 100%, with
no intermediate value, and the populated set is *exactly* the set of rows that
carry a producer:

```
total_terpenes populated == producer present:  33,229 / 33,229 rows, no exceptions
laboratories at 100%:  G3, NV CANN, DB, CERTIFIED AG, DPL NV, ERP, 374, MA
laboratories at   0%:  PureVita, Cannalytics RI, Lifted Testing, Green Peaks
```

So `total_terpenes` being blank is an exact in-model indicator of "this row will
be dropped by the unseen-producer scheme". That coupling is why the headline
comparison in 3.2 is made on the rows that name a producer.

Finally, 101 rows carry a `total_thc` above 100%, which is impossible as a
percentage; the largest is 686,400, a milligram figure in a percent column.
Ninety-four come from a single laboratory. They are 0.27% of the corpus and we
leave them in place rather than silently editing source measurements, but they
are unit artefacts rather than chemistry and no claim here rests on them.

## S5. Missingness patterns

The 4.1-point lift of the missingness-only probe (2.3) is not spread evenly. The
19 columns take only 13 distinct missingness patterns, and 7 of them are **100%
one class**:

**Table S6.** Missingness patterns that contain a single class, rows as stored;
the three largest of the seven are listed.

```
patterns that are 100% one class: 7 of 13
rows they cover:                  4123 (11.04%)
  n= 1495  all solventless (6 of 19 analytes reported)
  n= 1303  all hydrocarbon (1 of 19 analytes reported)
  n= 1303  all hydrocarbon (4 of 19 analytes reported)
```

For 11% of the usable rows the label is recoverable with certainty from which
cells are blank, before any number is read. Almost all of these rows, 4,115 of
4,123, come from four laboratories (`PureVita`, `Cannalytics RI`, `Lifted
Testing`, `Green Peaks`) that report a restricted analyte panel and record no
producer, so the shortcut is a laboratory reporting convention and not
chemistry. Much of it is an artefact of the split records of 2.1: the rows
reporting one analyte and the rows reporting four are the two halves of the same
1,303 samples (1,230 of them at `PureVita`), all solvent-based, while the rows
reporting six are all non-solvent. With each pair merged the columns take 12
patterns, 6 of them pure, covering 2,820 rows (7.82%), and the in-sample lookup
scores 68.8% against a 64.6% majority. The remaining 8 rows sit in three rare
patterns from other laboratories and do carry a producer; 14 of the pure rows
report no analyte at all and stay in the data. The 4,115 producer-less rows are
also the rows the unseen-producer scheme must drop, which is why the headline
comparison in 3.2 is made without them.

Laboratory and class are not degenerate, though they are far from independent.
Only 373 rows, 1.0%, sit in laboratories that are more than 90% one class, and
the two largest laboratories, 15,018 and 12,531 rows, both sit close to the
overall 65.8/34.2 split; across all twelve the hydrocarbon share runs from 0.0%
to 97.8%, and S9 gives the range for the eight held-out laboratories.

## S6. Learning rate of the alarm probes

The probes are multi-class, 11 and 33 classes, and the softmax boosting diverges
on them at scikit-learn's default learning rate of 0.1: the laboratory probe's
five folds score 40.6 to 69.4 and the producer probe's 10.4 to 68.7. At a
learning rate of 0.05 the fold scores span 0.6 points for the laboratory and 1.6
for the producer, and those are the figures reported in 3.1. The binary
extraction task is less sensitive and keeps the default: on all usable rows its
random split scores 81.0% at the default rate and 79.0% at 0.05, with fold
spreads of 0.4 and 0.5, and on the producer rows the held-out score is 54.0% and
54.1%.

## S7. Confusion matrix and per-class recall

The pooled confusion matrix of the reported model with the producer held out
(3.2) shows where its apparent skill was concentrated:

**Table S7.** Pooled confusion matrix of the reported model, producer held out.

```
unseen-producer     true hydrocarbon: 16,149 correct /  5,818 wrong
                    true solventless:  3,332 correct /  7,930 wrong
```

Solventless recall is **32.3%** as the mean of the five per-fold recalls and
3,332 / 11,262 = **29.6%** pooled. The two differ because the folds carry very
unequal numbers of solventless rows, from 951 to 4,413, and the fold with the
fewest has the highest recall (48.5% against 25.3% in the fold with the most).
The pooled matrix gives a balanced accuracy of 51.6%. For comparison, the
per-class recalls are 89.8% and 67.1% under the random split on the same rows,
and 69.9% and 39.3% on the dual-class subset grouped by producer. The model
mostly answers with the majority class, and its residual skill lives on the easy
side.

## S8. The strain key

Section 4.3 explains why strain could not be tested cleanly. The only cultivar
signal in the corpus is a key extracted from `product_name`, and Table S8 shows
what it contains:

**Table S8.** What a strain key extracted from product names contains.

```
usable rows                                     37,344
  no cultivar key extractable at all             4,717   (12.6%)
  key extracted                                 32,627   (87.4%)
    of those, an obvious non-cultivar word       7,386   (22.6% of keys)
    (oil, thc, lab sample, acres, ethanol, raw)
rows yielding no usable cultivar signal         12,103   (32.4% of usable)
```

The 4,717 rows with no key are the 4,115 producer-less rows, which have no
product name, and 602 named ones; nearly a quarter of the keys that do parse are
packaging or process words rather than cultivar names.

For completeness: grouping the method task by strain key gives 66.2% balanced
accuracy (+/-1.6), against 79.2% for a random split on the keyed rows, and
restricting to the 214 keys that appear in both classes, 9,598 rows, gives 56.9%
(+/-11.4), against 83.6%, with a macro F1 of 42.8%. An alarm probe in the sense
of 2.6, predicting the key from chemistry on the 109 keys with at least 30 rows
(12,829 rows), gives 32.4% against a 22.5% baseline, a lift of 9.9 points, where
the majority key is the non-cultivar word `oil`. We rest no conclusion on any of
the three: a 22.6%-contaminated grouping key makes the fold composition partly
arbitrary. The strain question is open and cannot be answered with this corpus
alone.

## S9. Holding out the laboratory

The `unseen-lab` scheme of 2.4 has the highest mean of the three group-held-out
schemes, 68.3%, and a fold-to-fold standard deviation of **18.6 points**: the
score depends on which laboratory is held out. Across the eight held-out
laboratories the hydrocarbon share ranges from 13.5% (`Cannalytics RI`, 758
rows) to 78.4% (`PureVita`, 3,137 rows), against an overall 65.8%, so holding
out a laboratory shifts the test-set prior as well as the instrument, and the
two effects cannot be separated with these data. With the split pairs of 2.1
merged the scheme scores 67.4% (+/-19.3). We report it for completeness and do
not rest any conclusion on it.
