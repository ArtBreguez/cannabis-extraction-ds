"""Probe strain as a confounder — using product_name, since strain_name is empty.

Phase 4 left one confounder untested: cultivar. The theory is that solventless
production favours strains that wash well, so the label may partly encode
"which cultivar", and cultivar drives terpenes directly.

The dedicated strain_name / strain_type columns are 100% empty for concentrate
rows (measured), so the only cultivar signal available is inside product_name
("blue dream (1g)", "kimbo cookies (1g)"). That is noisy, but it is what the
dataset offers, and a noisy proxy that still shows leakage is a stronger result
than no test.

Method: strip weight/size/texture tokens from product_name to get a coarse
strain key, keep strains that appear in BOTH classes with enough volume, then
score the method task grouped by that strain key. If holding out the strain
collapses performance the way holding out the producer did, cultivar is
confirmed as a dominant confounder.

Output: docs/evidence/phase4b_strain.txt
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data/labeled/master.csv"
OUT = ROOT / "docs/evidence/phase4b_strain.txt"
SEED = 20260921

# Tokens that describe packaging or texture, not cultivar. Stripped so that
# "blue dream 1g" and "blue dream live rosin" collapse to the same strain key.
NOISE = re.compile(
    r"\b(\d+(\.\d+)?\s?(g|mg|gram|grams|oz|ml)|live\s+rosin|hash\s+rosin|"
    r"rosin|live\s+resin|resin|bho|shatter|badder|batter|wax|distillate|"
    r"concentrate|cured|cold|fresh|frozen|bulk|each|inhalable|sugar|"
    r"diamonds?|sauce|crumble|budder|preroll|cartridge|disposable|"
    r"\(.*?\)|\[.*?\]|#\w+|1g|\bh\b|\bi\b|\bs\b)\b",
    re.IGNORECASE,
)


def strain_key(name: str) -> str:
    s = (name or "").lower()
    s = NOISE.sub(" ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def features(df: pd.DataFrame) -> list[str]:
    # `date_tested` matches the "_tested" suffix but is a date, not a mask.
    flags = [c for c in df.columns
             if c.endswith("_tested") and c != "date_tested"]
    cols = [c[: -len("_tested")] for c in flags]
    return sorted(c for c in cols
                  if c in df.columns and df[c].notna().sum() > 0)


def model():
    return HistGradientBoostingClassifier(
        max_iter=200, random_state=SEED, early_stopping=False)


def evaluate(X, y, groups, name, out):
    n = min(5, len(np.unique(groups)))
    if n < 2:
        out.append(f"{name}: too few groups")
        return
    bal, f1s = [], []
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = model().fit(X[tr], y[tr])
        p = clf.predict(X[te])
        bal.append(balanced_accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    if bal:
        out.append(f"{name:22} folds={len(bal)}  "
                   f"balanced_acc={100*np.mean(bal):>5.1f}% "
                   f"(+/-{100*np.std(bal):.1f})  "
                   f"macro_F1={100*np.mean(f1s):>5.1f}%")


def main() -> int:
    df = pd.read_csv(MASTER, low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    feats = features(df)

    df["strain_key"] = df["product_name"].fillna("").map(strain_key)
    n_labelled = len(df)
    n_empty = int((df["strain_key"] == "").sum())
    df = df[df["strain_key"] != ""]

    # Obvious non-cultivar keys, reported so the contamination rate in the
    # manuscript has a denominator. 4.3 quotes these three figures; keeping
    # them in the log stops the share being recomputed against the wrong
    # population (an earlier draft divided noise+empty by the key-bearing
    # rows only, which is not a share of anything).
    JUNK = {"oil", "thc", "lab sample", "acres", "ethanol",
            "acres solvent based", "terpene free non solvent", "raw"}
    n_junk = int(df["strain_key"].isin(JUNK).sum())

    out: list[str] = []
    out.append(f"labelled rows:                 {n_labelled}")
    out.append(f"  no cultivar key extractable: {n_empty} "
               f"({100*n_empty/n_labelled:.1f}%)")
    out.append(f"  key extracted:               {len(df)} "
               f"({100*len(df)/n_labelled:.1f}%)")
    out.append(f"    of those, obvious non-cultivar: {n_junk} "
               f"({100*n_junk/len(df):.1f}% of keys)")
    out.append(f"rows with no usable cultivar signal: {n_empty + n_junk} "
               f"({100*(n_empty+n_junk)/n_labelled:.1f}% of labelled)")
    out.append("")
    out.append(f"rows with a usable strain key: {len(df)}")
    out.append(f"distinct strain keys: {df['strain_key'].nunique()}")
    vc = df["strain_key"].value_counts()
    out.append("top strain keys:")
    for k, v in vc.head(8).items():
        out.append(f"  {v:5d}  {k[:44]}")
    out.append("")

    # Alarm probe: does the chemistry predict the strain key itself?
    out.append("=== ALARM: chemistry -> strain key (>=30 samples) ===")
    big = vc[vc >= 30].index
    sub = df[df["strain_key"].isin(big)]
    if sub["strain_key"].nunique() >= 2:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        Xs = sub[feats].astype(float).to_numpy()
        ys = sub["strain_key"].astype(str).to_numpy()
        base = pd.Series(ys).value_counts().max() / len(ys)
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        sc = cross_val_score(model(), Xs, ys, cv=cv, scoring="accuracy",
                             n_jobs=-1)
        out.append(f"strains={sub['strain_key'].nunique()} rows={len(sub)}  "
                   f"acc={100*sc.mean():.1f}%  baseline={100*base:.1f}%  "
                   f"lift={100*(sc.mean()-base):+.1f}pts")
    out.append("")

    # The method task, grouped by strain key.
    out.append("=== METHOD TASK, GROUPED BY STRAIN KEY ===")
    X = df[feats].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()
    g = df["strain_key"].astype(str).to_numpy()
    evaluate(X, y, g, "unseen-strain (all)", out)

    # Strains that appear in both classes: strain carries no class info there.
    per = df.groupby("strain_key")["label"].nunique()
    dual = set(per[per == 2].index)
    m = df["strain_key"].isin(dual)
    out.append("")
    out.append("=== STRAINS SPANNING BOTH CLASSES ===")
    out.append("same cultivar made both ways: strain cannot encode the label")
    out.append(f"dual-class strains: {len(dual)}   rows: {int(m.sum())}")
    if m.sum() > 1000:
        sub = df[m]
        Xd = sub[feats].astype(float).to_numpy()
        yd = sub["label"].astype(str).to_numpy()
        gd = sub["strain_key"].astype(str).to_numpy()
        bal = pd.Series(yd).value_counts()
        out.append(f"class balance: {bal.to_dict()}")
        evaluate(Xd, yd, gd, "dual-strain+unseen", out)

    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
