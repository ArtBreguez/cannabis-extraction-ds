# Group-aware validation collapses extraction-method classification in cannabis chemical data

**Arthur Breguez**
Independent researcher
ORCID: pending
Correspondence: arthurbreguez@gmail.com

Preprint. Code and evidence logs: `cannabis-extraction-ds`.

---

## Abstract

Published models that classify cannabis products from cannabinoid and terpene
profiles report near-perfect accuracy: above 95% for resin type, 100% for
chemovar class. These figures are obtained with random cross-validation or
Kennard-Stone partitioning over data with group structure, where replicate
clones, repeat producers and repeat laboratories span the train/test boundary.

We quantify the cost of that choice on a task where the group structure is
explicit and measurable. Using 37,374 concentrate certificates of analysis with
a producer-declared extraction-method label, a gradient-boosted classifier
separates solventless from hydrocarbon extracts at 81.0% balanced accuracy
under random five-fold cross-validation. Holding out the producer drops it to
54.0%, against a 50.0% chance level. The strictest available test, restricted
to the 55 producers that make both product classes so that producer identity
carries no label information, gives 54.6%.

The collapse replicates on an independent, designed experiment: 162 HPLC-assayed
samples across six varieties and three laboratory extraction methods fall from
56.7% under random folds to 38.9% under leave-one-variety-out, against 33.3%
chance. In both datasets the chemistry predicts the confounder better than the
target. The profile identifies the testing laboratory 15.3 points above its
baseline in market data, and the plant variety at 60.7% against a 16.7% chance
level in the controlled experiment.

We conclude that extraction method is not recoverable from the analyte panels in
public cannabis COAs for an unseen producer. We did not rerun the published
studies, so we make no claim about their chemical conclusions. We claim only
that their figures are obtained under the protocol that inflates by 27 points on
our data, and so do not estimate performance on a new producer or genotype.

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
[4, 5, 6, 7] and has been described specifically for chemometrics, where
Kiraly and Toth show that the Kennard-Stone selection algorithm and the
inappropriate handling of repeated measurements both introduce train/test
leakage [8]. The cannabis applications cited above use precisely these
procedures. Reference [2] grows 88 genotypes cloned in triplicate to obtain 264
plants, then partitions with Kennard-Stone without grouping the clones;
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

A secondary labelling path applies regular expressions to `product_name`
(`hash rosin`, `live rosin`, `rosin` for solventless; `live resin`, `bho`,
`shatter`, `badder`, `wax` for hydrocarbon). Where both paths produce a label
they agree on 4,254 of 4,284 rows, 99.3%. The 30 contradictory rows are
excluded.

The resulting dataset is 37,374 rows:

```
hydrocarbon   24,576   65.8%
solventless   12,798   34.2%
```

Grouping variables are populated as follows:

```
producer      33,259 rows (89.0%)   96 distinct
lab           37,374 rows (100.0%)  12 distinct
strain_name        0 rows (0.0%)     0 distinct
```

**Controlled data.** Zenodo record 13823859 [10] is a designed experiment rather
than market data: six named varieties, each extracted by three laboratory
methods (maceration, ultrasound, supercritical CO2) under varied
time, temperature and pressure, with cannabinoids quantified by HPLC. The
design is fully balanced at 54 samples per method and 27 per variety, 162 total,
with four features (CBD, CBG, CBN, THC) and no missing values.

### 2.2 Feature construction

The source file carries 35 analyte columns. Sixteen are empty for every labelled
row and are dropped rather than imputed, since imputing a never-populated column
fabricates a feature. Nineteen analytes remain.

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
d_limonene              75.7%   81.1%           40.8%   65.6%
beta_caryophyllene      75.7%   81.1%           47.1%   68.3%
alpha_pinene            75.7%   81.1%           34.3%   56.3%
total_terpenes          88.2%   89.4%           56.9%   74.5%
```

Terpene panels are requested at similar rates for both classes, a gap of
1 to 6 points. The large gap, 18 to 25 points, is in *detection*: hydrocarbon
extracts report a non-zero terpene value far more often. That is a property of
the extracts or of their reporting, not of which test was ordered.

### 2.3 The leakage pre-check

A coverage gap of that size is a candidate shortcut. If the pattern of which
terpenes were reported tracked the class, a classifier could score well while
reading the reporting convention rather than the chemistry. We measured this
before modelling rather than assuming it away.

A classifier given **only the missingness pattern**, with every measured value
discarded, reaches 65.8% accuracy against a 65.8% majority-class baseline: a
lift of **+0.0 points**. Adding laboratory identity as an explicit feature
reaches 67.8%, a lift of 2.0 points.

Laboratory and class are also not entangled. Only 373 rows, 1.0%, sit in
laboratories that are more than 90% one class. The two largest laboratories,
15,035 and 12,539 rows, both sit close to the overall 65.8/34.2 split.

The cheapest shortcuts therefore buy almost nothing, and any model that performs
well must do so on measured chemistry. This is what made the main experiment
worth running.

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
3. **unseen-lab**: grouped by laboratory, eight folds;
4. **dual-class + unseen**: restricted to the 55 producers that make *both*
   classes, grouped by producer.

Scheme 4 is the strongest available evidence. Inside that subset, knowing the
maker tells you nothing about the label by construction, so any lift must come
from the chemistry. It covers 28,927 rows, 77.4% of the labelled data, with an
18,470 / 10,457 class split.

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

Before the main task, three probes asked whether the chemistry predicts metadata
it should not. A profile that identifies the laboratory is a profile that
partially encodes instrumentation, calibration and reporting convention, which
would compromise any score obtained while train and test share laboratories.

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
producer leaves 54.0%, four points above coin-flipping. The 27-point gap is a
direct measurement of how much of that 81.0% was producer identity rather than
extraction chemistry.

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

Solventless recall is **32.3%**. The model mostly answers with the majority
class and its residual skill lives on the easy side. A single balanced-accuracy
number would have concealed this; per-class recall does not.

The `unseen-lab` scheme deserves separate comment. Its mean, 68.3%, is the
highest of the three honest schemes, but its fold-to-fold standard deviation is
**18.6 points**. Performance swings wildly with which laboratory is held out.
That is the signature of a model keyed to laboratory convention rather than to a
stable chemical difference, and it is consistent with the alarm in 3.1. A study
reporting only the mean of this scheme would claim 68.3% for something that is
not reproducible across laboratories.

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
identify extraction method for a producer the model has never seen. Strain and
producer variation dominate whatever method signal exists.

The **methodological claim** is general and, we think, the more useful one.
Cannabis chemical data has group structure, and the field's standard validation
protocol ignores it. Two independent datasets, one market and one designed, both
show double-digit collapses when the group is held out. In both, the chemistry
predicts the confounder better than the target.

This gives a concrete magnitude to the concern raised abstractly in [8]: in this
domain the inflation is 27 points on market data and 18 on a controlled
experiment, not a rounding error.

### 4.2 What this does not establish

**Not** that extraction method leaves no chemical trace. Controlled studies with
fixed genetics and fixed instrumentation measure method effects directly; this
work says nothing about those.

**Not** that a better model would fail. Only that these features do not
generalise across producers. A richer assay, a full terpene panel or an LC-MS
fingerprint, is a different experiment.

**Not** that the labels are wrong. They are source-declared and agree with the
name-derived path on 99.3% of rows where both exist.

**Not** a claim about the correctness of the conclusions in [1], [2] or [3].
Their chemical findings may well hold. What we show is that the reported
performance figures are obtained in a regime where we measure substantial
inflation, and so cannot be read as estimates of performance on a new producer,
genotype or laboratory.

### 4.3 The confounder we could not test

Strain is the obvious remaining candidate. Solventless production skews toward
cultivars that wash well, so the class label may partly encode which cultivar
was used, and cultivar drives terpene profile directly.

We could not test it. The `strain_name` and `strain_type` columns exist in the
source file but are empty for **0 of 37,374** labelled rows. The only remaining
cultivar signal is inside `product_name`, and extracting a strain key from it
yields 37.1% obvious non-cultivars: `oil`, `thc`, `lab sample`, `ethanol`,
`acres`. A probe built on a key that is one-third noise cannot distinguish
"cultivar leaks" from "my regex leaks".

We report this as a data-availability limit, not a finding. For completeness,
the run on the noisy key gave a *negative* lift for predicting the key from
chemistry, minus 3.0 points, which if anything hints that strain is a weaker
fingerprint here than laboratory was. We do not rely on it.

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
accuracy against a 50.0% chance level, and 54.6% on the strictest subset, where
a random split reports 81.0%. The same collapse appears in a designed HPLC
experiment when variety is held out, 56.7% to 38.9% against 33.3% chance. In
both datasets the chemistry reads the confounder better than the target,
identifying the laboratory 15.3 points above baseline and the variety at nearly
four times chance.

The field's published accuracies, 95% and 100%, are obtained under the protocol
that inflates by 27 points on our data. We do not claim their chemical
conclusions are wrong, and we did not rerun them. We claim their numbers do not
answer the question a reader will assume they answer. Group-aware validation is
not a refinement here. It changes the conclusion.

## Data and code availability

All code, including the scripts that produce every number above, and the raw
evidence logs for each run, are in the `cannabis-extraction-ds` repository.
Source data: Cannlytics `cannabis_results` [9] (CC-BY-4.0) and Zenodo 13823859
[10] (CC-BY-4.0). Neither dataset is redistributed; both are retrieved by
scripts in `src/ingestion/`. A single script,
`src/evaluation/rederive_paper_numbers.py`, recomputes the dataset facts in
this manuscript directly from the labelled dataset.

## Competing interests

The author produces cannabis-related content on social media under the handle
@HiddenTerps and has a commercial interest in the solventless extract category.
This work reports a negative result about distinguishing that category from
hydrocarbon extracts, which runs against that interest rather than supporting
it. No funding was received.

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
