"""Adversarial pass seven: attack the inference, not the arithmetic.

Six passes checked that the numbers match the source, that the prose is
consistent, that citations resolve and that the self-references hold. None
attacked the INFERENCE. A hostile referee with statistical training would go
straight for these, in this order:

A. The headline 81.0% -> 54.0% compares two schemes, but `random` runs on all
   37,344 usable rows while `unseen-producer` can only run on rows that HAVE a
   producer, which is 89%. If those two populations differ, the 27-point gap
   conflates the split scheme with the sample. This is the single most
   damaging possible objection and the paper does not address it.

B. Scheme 4 claims that inside the dual-class subset "knowing the maker tells
   you nothing about the label by construction". That is false as stated:
   making both classes does not mean making them in equal proportion. If
   producer A is 90% hydrocarbon and producer B is 20%, producer identity
   still carries label information. The claim needs to be either weakened or
   quantified.

C. 54.0% against a 50.0% chance level with a fold spread of 5.6 is reported as
   if the 4 points were real. No confidence interval, no test. A referee will
   ask whether 54.0% is distinguishable from chance at all.

D. Everything runs on one seed. A single-seed result for a stochastic
   estimator invites the obvious question of whether the gap survives
   reseeding.

Each of these is measured here. The point is not to defend the paper, it is to
find out which objections are fatal, which need a caveat, and which are
already fine.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260921
OUT = ROOT / "docs/evidence/review7_statistical.txt"


class _Tee:
    """Mirror stdout into the evidence log, so the figures 3.2 quotes from
    this pass (same-sample gap, producer-ID probe, interval, reseeds) have a
    committed source like every other run."""

    def __init__(self, path):
        self._fh = open(path, "w", encoding="utf-8")
        self._out = sys.stdout

    def write(self, s):
        self._out.write(s)
        self._fh.write(s)

    def flush(self):
        self._out.flush()
        self._fh.flush()


sys.stdout = _Tee(OUT)

df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
df = df[df["label_conflict"] == 0].copy()

flags = [c for c in df.columns
         if c.endswith("_tested") and c != "date_tested"]
analytes = sorted({c[: -len("_tested")] for c in flags})
feats = [a for a in analytes if a in df.columns and df[a].notna().sum() > 0]
cols = feats + [f"{a}_tested" for a in feats if f"{a}_tested" in df.columns]


def y_of(d):
    return (d["label"] == "solventless").astype(int).to_numpy()


def clf(seed=SEED):
    return HistGradientBoostingClassifier(max_iter=200, random_state=seed,
                                          early_stopping=False)


has_prod = df["producer"].notna() & (df["producer"].astype(str).str.len() > 0)
print(f"  rows usable               : {len(df):,}")
print(f"  rows with a producer      : {int(has_prod.sum()):,} "
      f"({100*has_prod.mean():.1f}%)")

# ---------- A. is the 27-point gap confounded by the sample? ----------
print("\n  === A. same-sample comparison: is the gap real? ===")
sub = df[has_prod]
X_sub = sub[cols].astype(float).to_numpy()
y_sub = y_of(sub)
g_sub = sub["producer"].astype(str).to_numpy()

rnd_all = cross_val_score(
    clf(), df[cols].astype(float).to_numpy(), y_of(df),
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    scoring="balanced_accuracy")
rnd_sub = cross_val_score(
    clf(), X_sub, y_sub,
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    scoring="balanced_accuracy")
grp_sub = cross_val_score(
    clf(), X_sub, y_sub, groups=g_sub, cv=GroupKFold(5),
    scoring="balanced_accuracy")

print(f"    random, ALL rows            {100*rnd_all.mean():.1f}% "
      f"(+/-{100*rnd_all.std():.1f})   <- what the paper reports as 81.0")
print(f"    random, producer rows only  {100*rnd_sub.mean():.1f}% "
      f"(+/-{100*rnd_sub.std():.1f})   <- the honest comparator")
print(f"    grouped, producer rows only {100*grp_sub.mean():.1f}% "
      f"(+/-{100*grp_sub.std():.1f})")
print(f"    gap as published            {100*(rnd_all.mean()-grp_sub.mean()):.1f} points")
print(f"    gap on a SINGLE sample      {100*(rnd_sub.mean()-grp_sub.mean()):.1f} points")

# ---------- B. does producer identity really carry no label info? ----------
print("\n  === B. the dual-class subset: 'no label information' ===")
lab = df.groupby(df["producer"].astype(str))["label"]
counts = lab.value_counts().unstack(fill_value=0)
counts = counts[(counts.get("solventless", 0) > 0)
                & (counts.get("hydrocarbon", 0) > 0)]
dual = counts.index.tolist()
d2 = df[df["producer"].astype(str).isin(dual)]
share = (counts["solventless"] / counts.sum(axis=1))
print(f"    dual-class producers        : {len(dual)}")
print(f"    rows                        : {len(d2):,}")
print(f"    solventless share per producer:")
print(f"      min {share.min():.2f}   median {share.median():.2f}   "
      f"max {share.max():.2f}")
print(f"      producers outside 0.2-0.8 : "
      f"{int(((share < 0.2) | (share > 0.8)).sum())} of {len(share)}")
# a classifier given ONLY the producer, inside the dual-class subset
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OrdinalEncoder
pid = OrdinalEncoder().fit_transform(
    d2["producer"].astype(str).to_numpy().reshape(-1, 1))
prod_only = cross_val_score(
    clf(), pid, y_of(d2),
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    scoring="balanced_accuracy")
print(f"    producer ID alone predicts label at "
      f"{100*prod_only.mean():.1f}% (+/-{100*prod_only.std():.1f})")
print("    -> if this is well above 50, the 'by construction' claim is wrong")

# ---------- C. is 54.0% distinguishable from chance? ----------
print("\n  === C. is the grouped score above chance at all? ===")
for name, scores in (("unseen-producer", grp_sub),):
    t, p = stats.ttest_1samp(scores, 0.5)
    lo, hi = stats.t.interval(0.95, len(scores) - 1, loc=scores.mean(),
                              scale=stats.sem(scores))
    print(f"    {name}: mean {100*scores.mean():.1f}%  "
          f"95% CI [{100*lo:.1f}, {100*hi:.1f}]  p={p:.3f}")
    print(f"      folds: {[round(100*float(s),1) for s in scores]}")
    print(f"      CI includes 50%: {lo <= 0.5 <= hi}")

# ---------- D. does the gap survive reseeding? ----------
print("\n  === D. three more seeds ===")
for s in (1, 7, 12345):
    r = cross_val_score(clf(s), X_sub, y_sub,
                        cv=StratifiedKFold(5, shuffle=True, random_state=s),
                        scoring="balanced_accuracy").mean()
    g = cross_val_score(clf(s), X_sub, y_sub, groups=g_sub,
                        cv=GroupKFold(5),
                        scoring="balanced_accuracy").mean()
    print(f"    seed {s:>6}: random {100*r:.1f}%  grouped {100*g:.1f}%  "
          f"gap {100*(r-g):.1f}")
