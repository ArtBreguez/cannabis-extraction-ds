"""Pass eleven, part B: is "54%" a property of the data or of the threshold?

The headline estimator is unweighted and the classes are roughly 1:2. Under
the producer hold-out it answers mostly with the majority class (solventless
recall 32%), so balanced accuracy at the default threshold can sit near 50
even when the scores rank the classes better than chance. A class-weighted
logistic regression reaching 61.1% held out (review11_checks.txt) says this
matters. Measured here on the producer rows and on the dual-class rows:

  - ROC AUC, which needs no threshold, random against producer-held-out
  - balanced accuracy with class_weight="balanced"
  - twenty repeated partitions and the producer bootstrap for the weighted
    model and for the AUC

Output: docs/evidence/review11b_weighting.txt
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.phase4_generalisation import SEED, features  # noqa: E402
from src.evaluation.review10b_robustness import linear_matrix  # noqa: E402

OUT = ROOT / "docs/evidence/review11b_weighting.txt"
POS = "solventless"


def hgb(cw=None):
    return HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                          early_stopping=False, class_weight=cw)


def logit(cw=None):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight=cw))


def cv_eval(make, X, y, splits):
    """Per-fold balanced accuracy and AUC, plus out-of-fold score and label."""
    bal, auc, rec = [], [], []
    score = np.full(len(y), np.nan)
    pred = np.empty(len(y), dtype=object)
    fold = np.full(len(y), -1)
    for k, (tr, te) in enumerate(splits):
        m = make().fit(X[tr], y[tr])
        pi = list(m.classes_).index(POS)
        s = m.predict_proba(X[te])[:, pi]
        p = m.predict(X[te])
        score[te], pred[te], fold[te] = s, p, k
        bal.append(balanced_accuracy_score(y[te], p))
        auc.append(roc_auc_score(y[te] == POS, s))
        rec.append(recall_score(y[te], p, average=None, labels=["hydrocarbon", POS]))
    return (np.array(bal), np.array(auc), np.mean(rec, axis=0),
            score, pred.astype(str), fold)


def boot(y, g, fn, n=2000):
    idx_by = defaultdict(list)
    for i, gg in enumerate(g):
        idx_by[gg].append(i)
    groups = list(idx_by)
    idx_by = {k: np.array(v) for k, v in idx_by.items()}
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        ii = np.concatenate([idx_by[groups[k]] for k in rng.choice(len(groups), len(groups))])
        if len(set(y[ii])) < 2:
            continue
        vals.append(fn(ii))
    v = np.array(vals)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return lo, hi, float(np.mean(v <= 0.5))


def f(a):
    return f"{100*a.mean():.1f}% (+/-{100*a.std():.1f})"


def main() -> int:
    df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    feats = features(df)
    prod = df["producer"].fillna("").astype(str).str.strip()
    has = (prod != "").to_numpy()
    per = df[has].groupby(prod[has])["label"].nunique()
    dual = has & prod.isin(set(per[per == 2].index)).to_numpy()
    Xall = df[feats].astype(float).to_numpy()
    XLall = linear_matrix(df, feats)
    yall = df["label"].astype(str).to_numpy()
    gall = prod.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = [f"rows: {len(df)}   features: {len(feats)}   positive class for AUC: {POS}"]

    def say(line=""):
        out.append(line)
        print(line, flush=True)

    for name, m in (("producer rows", has), ("dual-class rows", dual)):
        X, XL, y, g = Xall[m], XLall[m], yall[m], gall[m]
        say()
        say(f"=== {name}: n={len(y)} producers={len(set(g))} ===")
        rs = list(cv.split(X, y))
        gs = list(GroupKFold(5).split(X, y, g))
        for label, make, XX in (("boosting, unweighted", lambda: hgb(None), X),
                                ("boosting, class-weighted", lambda: hgb("balanced"), X),
                                ("logistic, unweighted", lambda: logit(None), XL),
                                ("logistic, class-weighted", lambda: logit("balanced"), XL)):
            rb, ra, rr, *_ = cv_eval(make, XX, y, rs)
            gb, ga, gr, score, pred, fold = cv_eval(make, XX, y, gs)
            say(f"{label:26} random: bal_acc {f(rb)}  AUC {ra.mean():.3f}   "
                f"held out: bal_acc {f(gb)}  AUC {ga.mean():.3f} (+/-{ga.std():.3f})  "
                f"recall hydro/solventless {100*gr[0]:.1f}/{100*gr[1]:.1f}%")
            yb = (y == POS)
            pooled_auc = roc_auc_score(yb, score)
            lo, hi, below = boot(y, g, lambda ii: roc_auc_score(yb[ii], score[ii]))
            say(f"{'':26} held out, pooled AUC {pooled_auc:.3f}  producer bootstrap 95% "
                f"[{lo:.3f}, {hi:.3f}]  draws at or below 0.5: {100*below:.1f}%")
            pooled_bal = balanced_accuracy_score(y, pred)
            lo, hi, below = boot(y, g, lambda ii: balanced_accuracy_score(y[ii], pred[ii]))
            say(f"{'':26} held out, pooled bal_acc {100*pooled_bal:.1f}%  producer bootstrap 95% "
                f"[{100*lo:.1f}, {100*hi:.1f}]  draws at or below 50.0%: {100*below:.1f}%")

        if name == "producer rows":
            means_b, means_a = [], []
            for s in range(20):
                sp = list(GroupKFold(5, shuffle=True, random_state=s).split(X, y, g))
                gb, ga, *_ = cv_eval(lambda: hgb("balanced"), X, y, sp)
                means_b.append(gb.mean()); means_a.append(ga.mean())
            mb, ma = np.array(means_b), np.array(means_a)
            say(f"boosting, class-weighted, 20 shuffled partitions: bal_acc min {100*mb.min():.1f}%  "
                f"median {100*np.median(mb):.1f}%  max {100*mb.max():.1f}%   "
                f"AUC min {ma.min():.3f}  median {np.median(ma):.3f}  max {ma.max():.3f}   "
                f"partitions with bal_acc at or below 50.0%: {int((mb <= 0.5).sum())}")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
