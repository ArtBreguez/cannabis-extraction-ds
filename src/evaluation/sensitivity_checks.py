"""Two robustness figures the manuscript states in 2.2 and 4.2, recomputed.

Both were measured during review and quoted in the text, but neither had a
committed log. The manuscript promises a raw log for every run it describes,
so this script produces it. The estimator, seed and feature set are those of
phase4_generalisation.py, imported rather than copied so they cannot drift.

  1. 4.2: drop the 64 `Infused Pre-Rolls (Non-Solvent)` rows, which are
     solventless by declaration but are not concentrates, and rerun the
     random and unseen-producer schemes.
  2. 2.2: drop `total_terpenes`, which is a near-exact sum of the ten terpene
     columns, and rerun the random scheme.
  3. 3.2: the dual-class scheme's five fold scores, with the same one-sample
     t-interval that review7_statistical.py applies to unseen-producer.

Output: docs/evidence/sensitivity_checks.txt
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.phase4_generalisation import SEED, features  # noqa: E402

OUT = ROOT / "docs/evidence/sensitivity_checks.txt"


def clf():
    return HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                          early_stopping=False)


def random_split(df, feats):
    X = df[feats].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()
    s = cross_val_score(clf(), X, y,
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        scoring="balanced_accuracy")
    return s


def grouped_split(df, feats, group):
    g = df[group].fillna("").astype(str).str.strip()
    m = (g != "").to_numpy()
    X = df[feats].astype(float).to_numpy()[m]
    y = df["label"].astype(str).to_numpy()[m]
    s = cross_val_score(clf(), X, y, groups=g.to_numpy()[m], cv=GroupKFold(5),
                        scoring="balanced_accuracy")
    return s


def fmt(s):
    return f"{100*s.mean():.1f}% (+/-{100*s.std():.1f})"


def main() -> int:
    df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    feats = features(df)
    out: list[str] = []
    out.append(f"rows: {len(df)}   features: {len(feats)}")

    out.append("")
    out.append("=== REFERENCE: the two schemes as reported in 3.2 ===")
    r_all = random_split(df, feats)
    g_all = grouped_split(df, feats, "producer")
    out.append(f"random (all rows)        {fmt(r_all)}")
    out.append(f"unseen-producer          {fmt(g_all)}")
    out.append(f"gap                      {100*(r_all.mean()-g_all.mean()):.1f} points")

    out.append("")
    out.append("=== 4.2: WITHOUT THE 64 PRE-ROLL ROWS ===")
    pre = df["product_type"].fillna("").str.lower().str.contains("pre-roll")
    out.append(f"pre-roll rows dropped: {int(pre.sum())} "
               f"{df.loc[pre, 'product_type'].value_counts().to_dict()}")
    d2 = df[~pre]
    r2 = random_split(d2, feats)
    g2 = grouped_split(d2, feats, "producer")
    out.append(f"random (all rows)        {fmt(r2)}")
    out.append(f"unseen-producer          {fmt(g2)}")
    out.append(f"gap                      {100*(r2.mean()-g2.mean()):.1f} points")

    out.append("")
    out.append("=== 2.2: WITHOUT total_terpenes ===")
    f3 = [f for f in feats if f != "total_terpenes"]
    r3 = random_split(df, f3)
    out.append(f"features: {len(f3)}")
    out.append(f"random (all rows)        {fmt(r3)}")
    out.append(f"cost of dropping it      {100*(r_all.mean()-r3.mean()):.1f} points")

    out.append("")
    out.append("=== 3.2: DUAL-CLASS SCHEME, FOLD SCORES AND INTERVAL ===")
    prod = df["producer"].fillna("").astype(str).str.strip()
    has = prod != ""
    per = df[has].groupby(prod[has])["label"].nunique()
    dual = set(per[per == 2].index)
    d4 = df[has & prod.isin(dual)]
    g4 = grouped_split(d4, feats, "producer")
    t, p = stats.ttest_1samp(g4, 0.5)
    lo, hi = stats.t.interval(0.95, len(g4) - 1, loc=g4.mean(),
                              scale=stats.sem(g4))
    out.append(f"producers: {len(dual)}   rows: {len(d4)}")
    out.append(f"dual-class+unseen        {fmt(g4)}")
    out.append(f"folds: {[round(100*float(v), 1) for v in g4]}")
    out.append(f"95% CI [{100*lo:.1f}, {100*hi:.1f}]  p={p:.3f}  "
               f"(one-sample t against 50.0%)")
    out.append(f"CI includes 50%: {lo <= 0.5 <= hi}")

    out.append("")
    out.append("=== 3.2: UNSEEN-PRODUCER, PER-FOLD SOLVENTLESS RECALL ===")
    out.append("the fold-averaged recall and the pooled matrix differ because")
    out.append("the folds carry very unequal numbers of solventless rows")
    from sklearn.metrics import recall_score
    sub = df[has]
    Xs = sub[feats].astype(float).to_numpy()
    ys = sub["label"].astype(str).to_numpy()
    gs = prod[has].to_numpy()
    recs, n_sol = [], []
    for k, (tr, te) in enumerate(GroupKFold(5).split(Xs, ys, gs)):
        pred = clf().fit(Xs[tr], ys[tr]).predict(Xs[te])
        r = recall_score(ys[te], pred, pos_label="solventless")
        n_s = int((ys[te] == "solventless").sum())
        recs.append(r)
        n_sol.append(n_s)
        out.append(f"fold {k}: test rows={len(te):>6}  solventless={n_s:>5}  "
                   f"solventless recall={100*r:.1f}%")
    out.append(f"mean of fold recalls: {100*np.mean(recs):.1f}%   "
               f"min {100*min(recs):.1f}%   max {100*max(recs):.1f}%")
    out.append(f"solventless rows per fold: min {min(n_sol)}   max {max(n_sol)}")

    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
