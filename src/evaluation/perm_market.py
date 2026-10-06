"""Permutation tests for the two producer-grouped market schemes.

3.3 reports a permutation p for the controlled data; a referee asked for the
same on the market data, where the one-sample t over five folds is weak.
Labels are shuffled globally, the grouped folds are kept, and the observed
mean balanced accuracy is compared with the null distribution.

Output: docs/evidence/perm_market.txt
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.phase4_generalisation import SEED, features  # noqa: E402

OUT = ROOT / "docs/evidence/perm_market.txt"
N_PERM = {"unseen-producer": 40, "dual-class+unseen": 60}


def score(X, y, splits):
    b = []
    for tr, te in splits:
        p = HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                           early_stopping=False).fit(X[tr], y[tr]).predict(X[te])
        b.append(balanced_accuracy_score(y[te], p))
    return float(np.mean(b))


def main() -> int:
    df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    feats = features(df)
    prod = df["producer"].fillna("").astype(str).str.strip()
    has = prod != ""
    per = df[has].groupby(prod[has])["label"].nunique()
    dual = set(per[per == 2].index)
    out = []
    for name, mask in (("unseen-producer", has),
                       ("dual-class+unseen", has & prod.isin(dual))):
        X = df.loc[mask, feats].astype(float).to_numpy()
        y = df.loc[mask, "label"].astype(str).to_numpy()
        g = prod[mask].to_numpy()
        splits = list(GroupKFold(5).split(X, y, g))
        obs = score(X, y, splits)
        n = N_PERM[name]
        null = np.array(Parallel(n_jobs=2)(
            delayed(score)(X, np.random.default_rng(s).permutation(y), splits)
            for s in range(n)))
        p = ((null >= obs).sum() + 1) / (n + 1)
        out.append(f"{name:18} obs={100*obs:.1f}%  perms={n}  null mean={100*null.mean():.1f}% "
                   f"sd={100*null.std():.1f}  95pct={100*np.percentile(null, 95):.1f}%  p={p:.3f}")
        print(out[-1], flush=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
