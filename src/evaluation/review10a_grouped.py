"""Pass ten, part A: how far above chance are the producer-held-out scores,
measured at the level where the data are actually independent?

A second referee pointed out that the row-level label shuffle used in pass
nine treats ~33k rows as exchangeable when labels are clustered in 96
producers, so its null (sd 0.1) is far too narrow, and that one deterministic
GroupKFold partition is a single draw. This script replaces both:

  A. 20 repeated GroupKFold partitions (shuffled producer-to-fold assignment)
     for the unseen-producer and dual-class schemes: the distribution of the
     mean balanced accuracy over partitions.
  B. A producer-cluster bootstrap of the pooled out-of-fold predictions from
     the reported partition: producers are resampled with replacement, so the
     interval reflects producer-to-producer variation.
  I. Laboratory identity alone, as a lookup under the same producer-grouped
     folds, to check that the few points above 50 are not a laboratory prior.

Output: docs/evidence/review10a_grouped.txt
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.phase4_generalisation import SEED, features  # noqa: E402

OUT = ROOT / "docs/evidence/review10a_grouped.txt"
N_PART = 20
N_BOOT = 2000


def clf():
    return HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                          early_stopping=False)


def cv_mean(X, y, splits):
    b, oof = [], np.empty(len(y), dtype=object)
    for tr, te in splits:
        p = clf().fit(X[tr], y[tr]).predict(X[te])
        b.append(balanced_accuracy_score(y[te], p))
        oof[te] = p
    return float(np.mean(b)), oof


def cluster_boot(y, oof, g, rng):
    idx_by = defaultdict(list)
    for i, gg in enumerate(g):
        idx_by[gg].append(i)
    groups = list(idx_by)
    idx_by = {k: np.array(v) for k, v in idx_by.items()}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        ii = np.concatenate([idx_by[groups[k]] for k in pick])
        if len(set(y[ii])) < 2:
            continue
        vals.append(balanced_accuracy_score(y[ii], oof[ii]))
    return np.array(vals)


def main() -> int:
    df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    feats = features(df)
    prod = df["producer"].fillna("").astype(str).str.strip()
    has = prod != ""
    per = df[has].groupby(prod[has])["label"].nunique()
    dual = set(per[per == 2].index)
    out = [f"rows: {len(df)}   features: {len(feats)}"]
    for name, mask in (("unseen-producer", has),
                       ("dual-class+unseen", has & prod.isin(dual))):
        X = df.loc[mask, feats].astype(float).to_numpy()
        y = df.loc[mask, "label"].astype(str).to_numpy()
        g = prod[mask].to_numpy()
        lab = df.loc[mask, "lab"].astype(str).to_numpy()
        out += ["", f"=== {name}: rows={len(y)} producers={len(set(g))} ==="]

        obs, oof = cv_mean(X, y, list(GroupKFold(5).split(X, y, g)))
        pooled = balanced_accuracy_score(y, oof.astype(str))
        out.append(f"reported partition: mean of folds {100*obs:.1f}%   "
                   f"pooled balanced_acc {100*pooled:.1f}%")

        means = []
        for s in range(N_PART):
            m, _ = cv_mean(X, y, list(GroupKFold(5, shuffle=True, random_state=s)
                                      .split(X, y, g)))
            means.append(m)
        means = np.array(means)
        out.append(f"A. {N_PART} shuffled partitions, mean of folds: "
                   f"min {100*means.min():.1f}%  median {100*np.median(means):.1f}%  "
                   f"max {100*means.max():.1f}%  mean {100*means.mean():.1f}%  "
                   f"sd {100*means.std(ddof=1):.1f}")
        out.append(f"   partitions at or below 50.0%: {int((means <= 0.5).sum())} of {N_PART}")
        out.append(f"   values: {[round(100*float(v), 1) for v in sorted(means)]}")

        boot = cluster_boot(y, oof.astype(str), g, np.random.default_rng(SEED))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(f"B. producer-cluster bootstrap of pooled balanced_acc ({len(boot)} draws): "
                   f"mean {100*boot.mean():.1f}%  95% interval [{100*lo:.1f}, {100*hi:.1f}]  "
                   f"share of draws at or below 50.0%: {100*(boot <= 0.5).mean():.1f}%")

        b = []
        for tr, te in GroupKFold(5).split(X, y, g):
            vote = defaultdict(Counter)
            for i in tr:
                vote[lab[i]][y[i]] += 1
            maj = Counter(y[tr]).most_common(1)[0][0]
            pred = np.array([vote[lab[i]].most_common(1)[0][0] if lab[i] in vote else maj
                             for i in te])
            b.append(balanced_accuracy_score(y[te], pred))
        out.append(f"I. laboratory identity alone, same folds: {100*np.mean(b):.1f}%")
        print("\n".join(out[-6:]), flush=True)

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
