"""Phase 5: the controlled experiment (Zenodo 13823859, Ecuador).

Different question from Cannlytics, and a cleaner one. This is a designed
experiment: 6 named cannabis varieties, each extracted by 3 lab methods
(maceration, ultrasound, supercritical CO2) under varied time/temp/pressure,
cannabinoids measured by HPLC. Genetics and instrument are controlled, which
is exactly what the market COA data could not offer.

Caveats stated up front, so the result is not oversold:
- These are LAB methods (maceration/ultrasound/supercritical), NOT the
  market categories the project cares about (rosin vs BHO). A signal here does
  not transfer to the solventless-vs-hydrocarbon question.
- Only 4 cannabinoid features (CBD, CBG, CBN, THC). No terpenes.
- 162 samples total. Small.

What makes it worth running anyway: it can answer the narrower question the
Cannlytics failure could not settle — CAN extraction method leave a
detectable chemical trace at all, when genetics are held fixed? The honest
test is variety-held-out: if method is separable only within a variety, it is
confounded; if it survives unseen varieties, the trace is real.

The data is long (648 rows = 162 samples x 4 cannabinoids). It is pivoted to
one row per sample with 4 cannabinoid columns.

Output: docs/evidence/phase5_controlled.txt
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/zenodo_13823859/BD2 canabinoide  flor hplc.xlsx"
OUT = ROOT / "docs/evidence/phase5_controlled.txt"
SEED = 20260921


def load_wide() -> pd.DataFrame:
    d = pd.read_excel(SRC, sheet_name="Rstudio")
    # R is the measured value; C names the cannabinoid. Everything else keys
    # the sample: method, variety, and the three process settings.
    keys = ["Metodo", "Variedad", "t", "T", "P"]
    d["R"] = pd.to_numeric(d["R"], errors="coerce")
    wide = d.pivot_table(index=keys, columns="C", values="R",
                         aggfunc="mean").reset_index()
    wide.columns.name = None
    for c in ["Metodo", "Variedad"]:
        wide[c] = wide[c].astype(str).str.strip()
    return wide


def model():
    return HistGradientBoostingClassifier(
        max_iter=200, random_state=SEED, early_stopping=False)


def score(X, y, splits, name, out):
    bal, f1s = [], []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 1:
            continue
        clf = model().fit(X[tr], y[tr])
        p = clf.predict(X[te])
        bal.append(balanced_accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    if bal:
        out.append(f"{name:24} folds={len(bal):>2}  "
                   f"balanced_acc={100*np.mean(bal):>5.1f}% "
                   f"(+/-{100*np.std(bal):.1f})  "
                   f"macro_F1={100*np.mean(f1s):>5.1f}%")
    else:
        out.append(f"{name}: no usable folds")


def main() -> int:
    df = load_wide()
    feats = ["CBD", "CBG", "CBN", "THC"]

    out: list[str] = []
    out.append(f"samples: {len(df)}   features: {feats}")
    out.append(f"methods: {df['Metodo'].value_counts().to_dict()}")
    out.append(f"varieties: {df['Variedad'].value_counts().to_dict()}")
    nan = df[feats].isna().sum().to_dict()
    out.append(f"missing per feature: {nan}")
    out.append("")

    X = df[feats].astype(float).to_numpy()
    y = df["Metodo"].to_numpy()
    n_classes = df["Metodo"].nunique()
    baseline = df["Metodo"].value_counts().max() / len(df)
    out.append(f"classes: {n_classes}  "
               f"majority baseline (plain acc): {100*baseline:.1f}%  "
               f"balanced-acc chance: {100/n_classes:.1f}%")
    out.append("")

    out.append("=== METHOD CLASSIFICATION ===")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    score(X, y, list(cv.split(X, y)), "random 5-fold", out)

    # The honest test: hold out an entire variety. If method is only separable
    # within a variety, this collapses; if it survives, the trace is real.
    logo = LeaveOneGroupOut()
    g = df["Variedad"].to_numpy()
    score(X, y, list(logo.split(X, y, g)), "leave-one-variety-out", out)

    out.append("")
    out.append("=== ALARM: does chemistry predict the VARIETY? ===")
    yv = df["Variedad"].to_numpy()
    bv = df["Variedad"].value_counts().max() / len(df)
    score(X, yv, list(StratifiedKFold(5, shuffle=True,
          random_state=SEED).split(X, yv)), "variety from chemistry", out)
    out.append(f"(variety majority baseline: {100*bv:.1f}%, "
               f"chance: {100/df['Variedad'].nunique():.1f}%)")

    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
