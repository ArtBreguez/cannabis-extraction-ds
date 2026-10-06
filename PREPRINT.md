# Group-aware validation collapses extraction-method classification in cannabis chemical data

**Arthur Gonçalves Breguez**
Independent researcher
ORCID: 0009-0005-8551-731X
Correspondence: arthurbreguez@gmail.com

Preprint. Code and evidence logs:
`https://github.com/ArtBreguez/cannabis-extraction-ds`

---

## Abstract

Published models that classify cannabis products from cannabinoid and terpene
profiles report near-perfect accuracy: above 95% for resin type, 100% for
chemovar class. These figures are obtained with random cross-validation or
Kennard-Stone partitioning over data with group structure, where replicate
clones, repeat producers and repeat laboratories span the train/test boundary.

We quantify the cost of that choice on a task where the group structure is
explicit and measurable. Using 37,374 cannabis certificates of analysis, almost
all of them concentrates, with a producer-declared extraction-method label, a
gradient-boosted classifier separates solventless from hydrocarbon extracts at
81.0% balanced accuracy under random five-fold cross-validation. Holding out the
producer drops it to 54.0%, a 95% confidence interval that contains the 50.0%
chance level. Restricted to the 53 producers making both classes, so a held-out
producer's label cannot be read off training rows, it gives 54.6%.

The collapse replicates on an independent, designed experiment: 162 HPLC-assayed
samples across six varieties and three laboratory extraction methods fall from
56.7% under random folds to 38.9% under leave-one-variety-out, against 33.3%
chance. In both datasets the chemistry predicts the confounder better than the
target. The profile identifies the testing laboratory 15.3 points above its
baseline in market data, and the plant variety at 60.7% against a 16.7% chance
level in the controlled experiment.

We conclude that extraction method is not recoverable from the analyte panels in
public cannabis COAs for an unseen producer. We did not rerun the published
studies and make no claim about their chemical conclusions, only that their
figures come from validation leaving group structure intact, the regime in which
we measure 27 points of inflation.

**Keywords:** cannabis, chemometrics, data leakage, group cross-validation,
extraction method, certificates of analysis, negative result

---

## 1. Introduction

Cannabis chemical profiles are now routinely fed to supervised classifiers.
Reported performance is consistently high. A support vector machine separates
Dutch-type from Moroccan-type cannabis resin at 95.2% accuracy from two
cannabinoid concentrations [1]. Partial least squares discriminant analysis
assigns chemovar class from near-infrared spectra of whole inflorescences with
"100% prediction accuracy", sensitivity and specificity both 1.00 [2]. A
combined cannabinoid-and-terpene PLS-DA model reaches 0% prediction
misclassification across chemovars [3].

These numbers invite a question the papers do not ask: performance on *what*
unseen unit? A shuffled fold defines "unseen" as a row absent from the training
partition. Deployment defines it more strictly, as a plant, a producer, or a
laboratory the model has never encountered. When several rows descend from the
same entity, a random split lets the model recognise the entity instead of
learning the property of interest, and the reported score measures memorisation.

This failure mode is established in the general methodological literature
[4, 5, 6, 7], with the earliest quantitative demonstration in genomics more
than two decades ago [11], and has been described specifically for chemometrics,
where Kiraly and Toth show that the Kennard-Stone selection algorithm and the
inappropriate handling of repeated measurements both introduce train/test
leakage [8]. The cannabis applications cited above use precisely these
procedures. Reference [2] grows 88 genotypes cloned in triplicate to obtain 264
plants, then partitions with Kennard-Stone [8] without grouping the clones;
reference [1] draws a random 80% for training and evaluates with plain accuracy
on a 149/40 class split, where always answering the majority class already
scores 78.8%.

The bridge between the two literatures has not been built. No published cannabis
study that we could locate holds out the producer, the genotype or the
laboratory and reports what happens to the score.

We build that bridge on a concrete task. The question is whether a concentrate's
cannabinoid and terpene profile carries enough information to recover how it was
made, separating solventless extracts (rosin, live rosin, hash rosin) from
hydrocarbon extracts (BHO, live resin, shatter, badder, wax). The task is a good
test case for three reasons. The label is declared at source rather than
inferred. The group structure is explicit, since every sample names a producer
and a laboratory. And a substantial subset of producers make both classes, which
permits a test where producer identity is provably uninformative.

We report a negative result for the chemical question and a quantitative result
for the methodological one.

## 2. Materials and methods

### 2.1 Data

**Market data.** The Cannlytics `cannabis_results` corpus [9] aggregates public
cannabis certificates of analysis obtained from state regulatory portals and
published COAs, released under CC-BY-4.0. The retrieved file contains 808,407
rows, of which 139,714 are concentrates or extracts.

Labels come from the `product_type` field, which in some markets declares the
extraction family directly as `non-solvent based concentrate` or
`solvent based concentrate`. This is a source-declared label, not an inference
from a commercial product name. Rows whose `product_type` is only `concentrate`
remain unlabelled and are never assigned to a class.

Most concentrate rows carry no such declaration and are therefore dropped at
this step, which is where the bulk of the reduction happens:

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
never recorded, so imputing one would invent the dependent variable. Matching
is by regular expression (`non[- ]?solvent`) rather than exact string, which
also admits regional variants such as `non-solvent concentrate` (1,445 rows)
and, in 64 cases, `infused pre-rolls (non-solvent)`. Those 64 rows are the
reason the labelled total slightly exceeds the declared count within the
concentrate subset; they are solventless by declaration but are pre-rolls
rather than concentrates, and they are 0.17% of the labelled data.

A secondary labelling path applies regular expressions to `product_name`
(`hash rosin`, `live rosin`, `rosin` for solventless; `live resin`, `bho`,
`shatter`, `badder`, `wax` for hydrocarbon). Where both paths produce a label
they agree on 4,254 of 4,284 rows, 99.3%. The 30 contradictory rows are
excluded.

The resulting labelled dataset is 37,374 rows, of which 30 carry contradictory
labels and are excluded from every analysis, leaving **37,344 rows** that all
figures below are computed on unless stated otherwise:

```
hydrocarbon   24,573   65.8%
solventless   12,771   34.2%
```

Grouping variables are populated as follows:

```
producer      33,229 rows (89.0%)   96 distinct
lab           37,344 rows (100.0%)  12 distinct
strain_name        0 rows (0.0%)     0 distinct
```

**Controlled data.** Zenodo record 13823859 [10] is a designed experiment rather
than market data: six named varieties, each extracted by three laboratory
methods (maceration, ultrasound, supercritical CO2) under varied
time, temperature and pressure, with cannabinoids quantified by HPLC. The
design is fully balanced at 54 samples per method and 27 per variety, 162 total,
with four features (CBD, CBG, CBN, THC) and no missing values.

### 2.2 Feature construction

The source file carries 34 analyte columns. Fifteen are empty for every
labelled row and are dropped rather than imputed, since imputing a
never-populated column fabricates a feature. Nineteen analytes remain.

Non-detects require care. The corpus has already normalised `ND` and `<LOQ`
strings away, so no such strings survive, which collapses a four-way
distinction into a two-way one:

- a numeric **zero** is a real measurement at or below the detection limit;
- an **empty** cell is an absence of information, either not tested or not
  reported.

These are not the same and must not be merged. Each analyte therefore
contributes two columns: the value, and a boolean `_tested` mask. Rows are never
dropped for having gaps.

This distinction produces two different and both meaningful coverage figures,
which we separate explicitly because conflating them is easy:

```
analyte               tested (flag)          detected (value > 0)
                   solventless  hydro      solventless  hydro
d_limonene              75.7%   81.1%           40.7%   65.6%
beta_caryophyllene      75.7%   81.1%           47.0%   68.3%
alpha_pinene            75.7%   81.1%           34.2%   56.3%
total_terpenes          88.2%   89.4%           56.8%   74.5%
```

Terpene panels are requested at similar rates for both classes, a gap of
1 to 6 points. The large gap, 18 to 25 points, is in *detection*: hydrocarbon
extracts report a non-zero terpene value far more often. That is a property of
the extracts or of their reporting, not of which test was ordered.

Two properties of these columns are worth stating because they bound what the
features can mean.

First, `total_terpenes` is not an independent measurement. It is a near-exact
sum of the ten terpenes that survive the empty-column drop (R² = 0.984 against
their sum, Pearson r = 0.992), so it is collinear with features already in the
matrix and any per-feature importance attributed to it is split credit rather
than chemistry. Its residual against that sum is laboratory-specific:
`CERTIFIED AG LAB` reports an exact sum on 99.7% of rows, `MA & ASSOCIATES` on
none, which makes the residual a laboratory fingerprint in its own right.
Dropping the column costs 0.3 points of balanced accuracy, so nothing in this
paper depends on keeping it; we keep it because the published literature does.

Second, whether `total_terpenes` is populated at all is a laboratory
convention, not a per-sample decision. Every laboratory is at 0% or 100%, with
no intermediate value, and the populated set is *exactly* the set of rows that
carry a producer:

```
total_terpenes populated == producer present:  33,229 / 33,229 rows, no exceptions
laboratories at 100%:  G3, NV CANN, DB, CERTIFIED AG, DPL NV, ERP, 374, MA
laboratories at   0%:  PureVita, Cannalytics RI, Lifted Testing, Green Peaks
```

So `total_terpenes` being blank is an exact in-model indicator of "this row
will be dropped by the unseen-producer scheme". That coupling is why 3.2
reports a same-sample comparison rather than resting on the headline gap.

Finally, 101 rows carry a `total_thc` above 100%, which is impossible as a
percentage; the largest is 686,400, a milligram figure in a percent column.
Ninety-four come from a single laboratory. They are 0.27% of the corpus and we
leave them in place rather than silently editing source measurements, but they
are unit artefacts rather than chemistry and no claim here rests on them.

### 2.3 The leakage pre-check

A detection gap of 18 to 25 points is a candidate shortcut. If the pattern of
which terpenes were reported, or reported as non-zero, tracked the class, a
classifier could score well while reading the reporting convention rather than
the chemistry. We measured this before modelling rather than assuming it away.

A classifier given **only the missingness pattern**, with every measured value
discarded, reaches 69.9% accuracy against a 65.8% majority-class baseline: a
lift of **+4.1 points**. Adding laboratory identity as an explicit feature
reaches 70.5%, a further 0.6 points.

The probe sees exactly the 19 analyte columns the model sees, which matters:
an earlier version of it looked at 6 terpenes only and reported a +0.0-point
lift, understating the available shortcut by four points. A probe narrower
than the feature set is a lower bound on the wrong quantity.

That 4.1-point lift is not spread evenly. The 19 columns take only 13 distinct
missingness patterns, and 7 of them are **100% one class**:

```
patterns that are 100% one class: 7 of 13
rows they cover:                  4123 (11.03%)
  n= 1495  all solventless (6 of 19 analytes reported)
  n= 1303  all hydrocarbon (1 of 19 analytes reported)
  n= 1303  all hydrocarbon (4 of 19 analytes reported)
```

For 11% of the corpus the label is recoverable with certainty from which cells
are blank, before any number is read. Almost all of these rows, 4,115 of 4,123,
come from four laboratories (`PureVita`, `Cannalytics RI`, `Lifted Testing`,
`Green Peaks`) that report a restricted analyte panel and record no producer,
so the shortcut is a laboratory reporting convention and not chemistry. The
remaining 8 rows sit in three sparsely-populated patterns from other
laboratories and do carry a producer. Those 4,115 rows are therefore also the
rows the unseen-producer scheme must drop, which is why the same-sample
comparison in 3.2 matters.

Laboratory and class are also not entangled. Only 373 rows, 1.0%, sit in
laboratories that are more than 90% one class. The two largest laboratories,
15,035 and 12,539 rows, both sit close to the overall 65.8/34.2 split.

The cheapest shortcut therefore buys 4.1 points, concentrated in 11% of the
rows and attributable to reporting convention. That is small enough that a
model scoring in the eighties cannot be running on it alone, which is what
made the main experiment worth running, but it is not nothing, and 3.2
reports what happens when those rows are excluded.

### 2.4 Model and split schemes

A single model is used throughout: `HistGradientBoostingClassifier`
(scikit-learn), `max_iter=200`, fixed seed, early stopping disabled. The point
of the study is the evaluation protocol, not the estimator, so the estimator is
held constant across all schemes.

Four split schemes are compared on the market data:

1. **random**: stratified five-fold, the protocol used in the cannabis
   literature;
2. **unseen-producer**: `GroupKFold` grouped by producer, so no producer
   appears in both train and test;
3. **unseen-lab**: leave-one-lab-out. Each fold holds out one entire
   laboratory. Only laboratories with at least 500 rows are eligible to be
   held out, because a fold testing on a 100-row laboratory estimates nothing;
   8 of the 12 laboratories clear that floor and all 8 contain both classes,
   giving 8 folds. The 4 excluded laboratories hold 561 rows, 1.50% of the
   data, and they are still present in every training set;
4. **dual-class + unseen**: restricted to the 53 producers that make *both*
   classes, grouped by producer.

Scheme 4 is the strongest available evidence, though not for the reason a
first reading suggests. Restricting to producers who make both classes removes
the degenerate case where a producer's identity determines its label outright,
but it does not make producer identity uninformative: the solventless share
across these 53 producers ranges from 0.00 to 0.99 with a median of 0.35, and
30 of the 53 sit outside the 0.2 to 0.8 band. A classifier given *only* an
encoded producer ID, and no chemistry at all, still reaches 84.4% balanced
accuracy inside this subset under a random split. What the restriction
guarantees is that the held-out producer's label cannot be inferred from
having seen that same producer in training, since every producer in the subset
contributes both classes. It covers 27,751 rows, 74.3% of the usable data,
with a 17,312 / 10,439 class split.

On the controlled data, random five-fold is compared against
leave-one-variety-out.

### 2.5 Metrics

The classes are 1:2, so plain accuracy is not reported. We use balanced accuracy
(chance 50.0% for two classes, 33.3% for three), macro F1, per-class recall,
confusion matrices, and the standard deviation across folds. Two baselines are
computed explicitly: most-frequent (balanced accuracy 50.0%) and stratified
random (50.7%).

Per-class recall matters here because a single aggregate figure hides which side
of the problem the model actually solves.

### 2.6 Alarm models

Before the main task, two probes asked whether the chemistry predicts metadata
it should not: the testing laboratory and the producer. A profile that
identifies the laboratory is a profile that partially encodes instrumentation,
calibration and reporting convention, which would compromise any score obtained
while train and test share laboratories. A third target, cultivar, has no
reliable field in this corpus and could only be probed on a key reconstructed
from product names; that probe and its limits are reported in 4.3.

Each probe keeps only classes with enough rows to be estimable, since a class
with 20 members contributes noise rather than signal: at least 100 rows for the
laboratory probe and 150 for the producer probe, the latter higher because
producers are far more numerous. These floors, applied after the 30 conflicting
rows are dropped, are why the probes in 3.1 run on 11 of 12 laboratories
(37,276 rows) and 33 of 96 producers (30,416 rows) rather than on the full
dataset. The `*_tested` flags are excluded from these probes, so that
"the chemistry identifies the laboratory" is not conflated with
"the missingness pattern identifies the laboratory", which 2.3 measures
separately.

## 3. Results

### 3.1 The alarm fires first

```
target      classes   rows     accuracy   baseline     lift
lab              11   37,276      55.6%      40.3%   +15.3 pts   [ALARM]
producer         33   30,416      24.3%      14.4%    +9.9 pts   [watch]
```

The chemical profile does identify the testing laboratory, 15.3 points above
its baseline. In hindsight this is unsurprising, since instruments, calibration
and reporting conventions differ between laboratories. Its consequence is not
optional: any score obtained while train and test share laboratories is partly
measuring the instrument.

### 3.2 Market data: 81.0% becomes 54.0%

```
split scheme            balanced_acc    macro_F1   folds
most-frequent baseline       50.0%           --      --
stratified baseline          50.7%           --      --
random (optimistic)          81.0% (+/-0.4)   81.8%      5
unseen-producer              54.0% (+/-5.6)   50.3%      5
unseen-lab                   68.3% (+/-18.6)  67.6%      8
dual-class + unseen          54.6% (+/-2.6)   50.7%      5
```

Under the protocol used in the published literature, the chemistry appears to
separate solventless from hydrocarbon extracts at 81.0%. Holding out the
producer leaves 54.0%, four points above coin-flipping.

Two qualifications are needed before that 27-point gap can be read as a pure
protocol effect, and both are measurable.

First, the two schemes do not run on the same rows: `random` uses all 37,344
usable rows while `unseen-producer` requires a producer and so is confined to
the 33,229 rows that have one, 89.0%. Comparing like with like, a random split
*restricted to those same rows* scores 78.5%, so the protocol-only gap is
**24.4 points** and the remaining 2.5 come from the easier sample. The paper's
headline figure of 27 is the comparison a reader of the published literature
would make, since those studies report the random-split number on all their
data; 24.4 is the stricter quantity and we report both.

Second, the grouped result is not distinguishable from chance by a
conventional test. Across the five producer-held-out folds the scores are
48.6, 57.6, 46.7, 55.7 and 61.5, giving a 95% confidence interval of
[46.3, 61.8] that contains 50.0% (one-sample t against chance, p = 0.22). The
honest reading is therefore not "54.0% is a small real effect" but "the
producer-held-out performance is statistically indistinguishable from
guessing". That strengthens rather than weakens the conclusion: the collapse
goes all the way to chance.

The gap is stable under reseeding. Three further seeds give protocol-only gaps
of 24.6, 24.5 and 24.6 points, with the grouped score at 54.0% in every case.

The strictest test agrees: 54.6% on the dual-class subset, with a tight
fold spread of 2.6 points. Macro F1 is 50.3% and 50.7% on the two
group-held-out schemes, against a 50.0% chance level, with fold spreads of 5.6
and 2.6. No gate that requires beating chance by more than the between-fold
spread is passed.

The confusion matrices show where the apparent skill was concentrated:

```
unseen-producer     true hydrocarbon: 16,149 correct /  5,818 wrong
                    true solventless:  3,332 correct /  7,930 wrong
```

Solventless recall is **32.3%** as the mean of the five per-fold recalls, the
figure the fold-averaged table above is built from. Pooling the matrix instead
gives 3,332 / 11,262 = **29.6%**; the two differ because the folds are of very
unequal size and the smallest carries the highest recall (48.5% against 25.3%
in the worst). We quote both so the matrix and the percentage can be
reconciled. Either way the model mostly answers with the majority
class and its residual skill lives on the easy side. A single
balanced-accuracy number would have concealed this; per-class recall does not.

The `unseen-lab` scheme deserves separate comment. Its mean, 68.3%, is the
highest of the three honest schemes, but its fold-to-fold standard deviation is
**18.6 points**. Performance swings wildly with which laboratory is held out.
That is the signature of a model keyed to laboratory convention rather than to a
stable chemical difference, and it is consistent with the alarm in 3.1. A study
reporting only the mean of this scheme would claim 68.3% for something that is
not reproducible across laboratories.

The per-laboratory class balance shows why the spread is so large. Across the
eight held-out laboratories the hydrocarbon share ranges from 13.5%
(`Cannalytics RI`, 758 rows) to 78.4% (`PureVita`, 3,137 rows), against an
overall 65.8%. Holding out a laboratory therefore shifts the test-set prior as
well as the instrument, and the two effects cannot be separated with this data.
We report the scheme for completeness and do not rest any conclusion on it.

### 3.3 Controlled experiment: the same collapse

```
split scheme              balanced_acc    macro_F1   folds
chance level (3 classes)       33.3%           --      --
random 5-fold                  56.7% (+/-9.2)   55.7%      5
leave-one-variety-out          38.9% (+/-12.6)  35.9%      6
```

A random split suggests method is somewhat separable at 56.7%. Holding out an
entire variety gives **38.9%**, 5.6 points above chance with a fold spread of
12.6 that straddles it. The same collapse as in the market data, with the
confounder changed from producer to variety.

And the same inversion:

```
chemistry -> variety:  balanced_acc 60.7% (+/-8.3)   chance 16.7%
```

The four cannabinoids predict the **variety** at nearly four times chance, far
better than they predict the extraction method they were measured to compare.
Even in an experiment built specifically to isolate method, with fixed genetics
per group and clean HPLC numbers, genetics dominates the cannabinoid signal.

This matters for interpretation. The Cannlytics result alone could be dismissed
as a consequence of messy market data. The controlled experiment removes that
explanation.

## 4. Discussion

### 4.1 What this establishes

Two claims, of different kinds.

The **chemical claim** is narrow: the analyte panels present in public cannabis
COAs, 19 analytes and heavily right-censored, do not carry enough information to
identify extraction method for a producer the model has never seen. Producer
variation dominates whatever method signal exists: inside the dual-class subset
a classifier given only an encoded producer ID reaches 84.4% while the chemistry
grouped by producer reaches 54.6%. Cultivar plausibly does the same, but the two
datasets disagree on the evidence and we do not claim it for the market corpus:
in the controlled experiment the four cannabinoids read variety at 60.7% against
16.7% chance while reading method at 38.9% against 33.3%, whereas in the market
data the chemistry does not read our reconstructed strain key at all, a lift of
minus 3.0 points (4.3).

The **methodological claim** is general and, we think, the more useful one.
Cannabis chemical data has group structure, and the field's standard validation
protocol ignores it. Two independent datasets, one market and one designed, both
show double-digit collapses when the group is held out. In both, the chemistry
predicts the confounder better than the target.

This gives a concrete magnitude to the concern raised abstractly in [8]: in this
domain the inflation is 27 points on market data, 24.4 of which survive a
same-sample comparison, and 18 on a controlled experiment, not a rounding
error.

### 4.2 What this does not establish

**Not** that extraction method leaves no chemical trace. Controlled studies with
fixed genetics and fixed instrumentation measure method effects directly; this
work says nothing about those.

**Not** that a better model would fail. Only that these features do not
generalise across producers. A richer assay, a full terpene panel or an LC-MS
fingerprint, is a different experiment.

**Not** that the labels are wrong. They are source-declared and agree with the
name-derived path on 99.3% of rows where both exist.

**Not** sensitive to the 64 pre-roll rows. Removing them moves random-split
balanced accuracy from 81.0% to 80.6% and leaves unseen-producer unchanged at
54.0%, so the 27-point gap becomes 26.6. The conclusion does not depend on
their inclusion.

**Not** a claim about the correctness of the conclusions in [1], [2] or [3].
Their chemical findings may well hold. What we show is that the reported
performance figures are obtained in a regime where we measure substantial
inflation, and so cannot be read as estimates of performance on a new producer,
genotype or laboratory.

### 4.3 The confounder we could not test cleanly

Strain is the obvious remaining candidate. Solventless production skews toward
cultivars that wash well, so the class label may partly encode which cultivar
was used, and cultivar drives terpene profile directly.

We could not test it cleanly. The `strain_name` and `strain_type` columns exist
in the source file but are empty for **0 of 37,344** usable rows. The only
remaining cultivar signal is inside `product_name`, and extracting a strain key
from it leaves a key too dirty to carry a confounder probe:

```
usable rows                                     37,344
  no cultivar key extractable at all             4,717   (12.6%)
  key extracted                                 32,627   (87.4%)
    of those, an obvious non-cultivar word       7,386   (22.6% of keys)
    (oil, thc, lab sample, acres, ethanol, raw)
rows yielding no usable cultivar signal         12,103   (32.4% of usable)
```

So roughly a third of the usable corpus carries no cultivar signal, and
nearly a quarter of the keys that do parse are packaging or process words
rather than cultivar names. A probe built on a key that dirty cannot
distinguish "cultivar leaks" from "my regex leaks".

We ran it anyway, and report the output rather than withholding it. Two of the
figures are grouped method results: grouping the method task by strain key
gives 66.2% balanced accuracy (+/-1.6), and restricting to the 214 keys that
appear in both classes, 9,598 rows, gives 56.9% (+/-11.4) with a macro F1 of
42.8%, below the 50.0% chance level. A third figure is a different quantity, an
alarm probe in the sense of 2.6: predicting the key itself from chemistry gave
a *negative* lift, minus 3.0 points, which if anything hints that strain is a
weaker fingerprint here than laboratory was.

Both grouped figures fall below the 81.0% random split, and the dual-class one,
at 6.9 points above chance, lands near the 4.0 and 4.6 points that the
producer-grouped schemes reach. The unrestricted strain grouping is the
exception: 66.2% sits 16.2 points above chance, four times the producer gap, so
it does not collapse the way the producer schemes do and we do not present it
as though it did. We rest no conclusion on any of the three, because a
22.6%-contaminated grouping key makes the fold composition partly arbitrary,
and a macro F1 below chance with an 11.4-point spread is instability rather
than signal. The honest status of the strain question is open and unanswerable
with this corpus alone.

### 4.4 Recommendations

For anyone fitting a classifier to cannabis chemical data:

1. **Group by the entity production will hand you fresh.** Producer, genotype,
   laboratory. `GroupKFold` and `LeaveOneGroupOut` cost nothing to run.
2. **Report the random-split score alongside the grouped one.** The gap is the
   informative quantity and is cheap to obtain.
3. **Never report plain accuracy on an imbalanced class.** Reference [1]'s
   95.2% sits on a 78.8% majority baseline, so the lift is 16.4 points.
4. **Report per-class recall and the between-fold spread.** Our `unseen-lab`
   mean of 68.3% carries a standard deviation of 18.6; the mean alone is
   misleading.
5. **Run an alarm model before the real task.** If the chemistry predicts the
   laboratory, any laboratory-sharing score is partly instrumental.
6. **Keep a tested-versus-detected distinction in censored panels.** Merging a
   measured zero with a missing cell changes the coverage gap in our data from
   roughly 5 points to roughly 22.

### 4.5 Where a decisive answer would come from

The remaining untested avenue is a method-comparison dataset pairing extraction
method with a full metabolomic fingerprint including terpenes, such as the
MassIVE LC-MS/GC-MS collections. That alone could distinguish "method leaves no
usable trace" from "method leaves a trace, but not in the 4-cannabinoid or
19-analyte projections tested here". It is a mass-spectrometry corpus of several
gigabytes and a separate build, not an extension of this work.

## 5. Conclusion

Extraction method is not identifiable from the cannabinoid and terpene panels in
public cannabis certificates of analysis for an unseen producer: 54.0% balanced
accuracy whose 95% confidence interval contains the 50.0% chance level, and
54.6% on the strictest subset, where a random split reports 81.0%. The same
collapse appears in a designed HPLC experiment when variety is held out, 56.7%
to 38.9% against 33.3% chance. In both datasets the chemistry reads the
confounder better than the target, identifying the laboratory 15.3 points above
baseline and the variety at nearly four times chance.

The field's published accuracies, 95% and 100%, are obtained under validation
that leaves the group structure intact, the family of protocols for which we
measure 27 points of inflation on market data, 24.4 of them on a same-sample
comparison, and 18 on a designed experiment. We ran random cross-validation,
not Kennard-Stone selection, so the magnitude we report is specific to the
former; what the two share is that neither holds out the producer, the genotype
or the laboratory. We do not claim the published chemical conclusions are
wrong, and we did not rerun those studies. We claim their numbers do not answer
the question a reader will assume they answer. Group-aware validation is not a
refinement here. It changes the conclusion.

## Data and code availability

Source data are public and are not redistributed here. The Cannlytics
`cannabis_results` corpus [9] and Zenodo record 13823859 [10] are both
CC-BY-4.0 and are retrieved by the ingestion scripts described below, which
verify the downloaded size against the figure declared by the host API.

All analysis code and the raw evidence log for every run described in this
manuscript are public at
`https://github.com/ArtBreguez/cannabis-extraction-ds` (code MIT, manuscript
CC-BY-4.0). The structure is:

```
src/ingestion/      retrieval with download-integrity verification
src/preprocessing/  labelling, unit checks, non-detect encoding
src/models/         the four split schemes and the alarm probes
src/evaluation/     recomputation and audit of every figure reported here
docs/evidence/      raw stdout of each run, committed unedited
```

A single script, `src/evaluation/rederive_paper_numbers.py`, recomputes every
dataset figure in this manuscript directly from the labelled dataset, and
`src/evaluation/audit_preprint.py` checks each claimed number in this text
against either that recomputation or the committed log, failing on any
disagreement.

## Competing interests

The author produces cannabis-related content on social media under the handle
@HiddenTerps and has a commercial interest in the solventless extract category.
This work reports a negative result about distinguishing that category from
hydrocarbon extracts, which runs against that interest rather than supporting
it. No funding was received.

## Use of AI tools

AI coding assistants were used throughout this work, under the author's
direction and review. Their contribution was to the analysis code, the
verification harness and the manuscript prose: writing and refactoring the
scripts in `src/`, running the audit passes described above, and drafting and
revising text that the author then checked against the evidence logs. Several
defects corrected before posting were found by that review loop, among them a
leakage pre-check that measured six analyte columns while the model saw
nineteen, and a percentage whose numerator and denominator described different
populations.

AI tools do not meet authorship criteria and are not listed as authors. The
author is responsible for every claim, number and conclusion in this
manuscript. Every reported figure is recomputed from the raw data by
`src/evaluation/rederive_paper_numbers.py` and asserted against this text by
`src/evaluation/audit_preprint.py`, which exits non-zero on any disagreement,
so no published value rests on an assistant's unverified output.

## A note on provenance

An early version of this dataset audit concluded the research was not viable,
reporting 6,336 rows and 24 versus 11 labellable samples. That measurement was
made on a download truncated to 7% of its size, 177.9 MB of 2,534.8 MB, cut
mid-record. `curl` reported success, the parser opened the file and discarded a
single malformed line, and every downstream number was plausible. The truncation
was found by comparing the local file size against the figure declared by the
host API. Every number in this manuscript comes from the verified intact file,
and download integrity is checked against the source in `src/ingestion/`.

## References

[1] Applying machine learning to international drug monitoring: classifying
cannabis resin collected in Europe using cannabinoid concentrations. *Journal of
Cannabis Research*, 2025. PMC11910419.

[2] Rapid In Situ Near-Infrared Assessment of Tetrahydrocannabinolic Acid in
Cannabis Inflorescences before Harvest Using Machine Learning. *Sensors*, 24(16),
5081, 2024. doi:10.3390/s24165081.

[3] Multivariate classification of cannabis chemovars based on their terpene and
cannabinoid profiles. *Phytochemistry*, 2022. doi:10.1016/j.phytochem.2022.113215.

[4] Kaufman, S., Rosset, S., Perlich, C., Stitelman, O. Leakage in data mining:
formulation, detection, and avoidance. *ACM TKDD*, 2012.

[5] Kapoor, S., Narayanan, A. Leakage and the reproducibility crisis in
machine-learning-based science. *Patterns*, 2023.

[6] Roberts, D. R., et al. Cross-validation strategies for data with temporal,
spatial, hierarchical, or phylogenetic structure. *Ecography*, 2017.

[7] Some combinatorics of data leakage induced by clusters. *Stochastic
Environmental Research and Risk Assessment*, 2024.
doi:10.1007/s00477-024-02715-1.

[8] Kiraly, P., Toth, G. Being Aware of Data Leakage and Cross-Validation
Scaling in Chemometric Model Validation. *Journal of Chemometrics*, 2025.
doi:10.1002/cem.70026.

[9] Cannlytics. `cannabis_results`: public cannabis lab test results.
Hugging Face Datasets. CC-BY-4.0.

[10] Zenodo record 13823859: cannabinoid quantification across extraction
methods and varieties. CC-BY-4.0. doi:10.5281/zenodo.13823859.

[11] Ambroise, C., McLachlan, G. J. Selection bias in gene extraction on the
basis of microarray gene-expression data. *PNAS*, 2002.
