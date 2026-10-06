"""Pass thirteen: four small figures a fifth referee asked for.

  1. Table 6 (tested / detected shares) on the rows that name a producer:
     on all usable rows it mixed in the producer-less laboratories, which
     never report terpenes, and the samples they store twice
  2. a random five-fold split on the same dated producer rows the temporal
     split uses, so that split has a like-for-like baseline
  3. the share of producer rows stamped Nevada
  4. how many of the rows with no extractable strain key simply have no
     product name

Output: docs/evidence/review13_checks.txt
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.phase4_generalisation import SEED, features  # noqa: E402
from src.models.phase4b_strain import strain_key  # noqa: E402

OUT = ROOT / "docs/evidence/review13_checks.txt"


def main() -> int:
    df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    feats = features(df)
    prod = df["producer"].fillna("").astype(str).str.strip()
    has = prod != ""
    P = df[has]
    out = [f"usable rows: {len(df)}   rows that name a producer: {len(P)}"]

    out += ["", "=== 1. tested / detected shares on the producer rows ===",
            f"{'analyte':22} {'tested':>18} {'detected (value > 0)':>24}",
            f"{'':22} {'solventless  hydro':>18} {'solventless  hydro':>24}"]
    for a in ("d_limonene", "beta_caryophyllene", "alpha_pinene", "total_terpenes"):
        t, d = {}, {}
        for cls in ("solventless", "hydrocarbon"):
            sub = P[P["label"] == cls]
            t[cls] = 100 * sub[a].notna().mean()
            d[cls] = 100 * (sub[a].fillna(0) != 0).mean()
        out.append(f"{a:22} {t['solventless']:>9.1f}% {t['hydrocarbon']:>6.1f}% "
                   f"{d['solventless']:>15.1f}% {d['hydrocarbon']:>6.1f}%")

    out += ["", "=== 2. random five-fold on the dated producer rows ==="]
    dt = pd.to_datetime(P["date_tested"], errors="coerce")
    D = P[dt.notna().to_numpy()]
    X = D[feats].astype(float).to_numpy()
    y = D["label"].astype(str).to_numpy()
    b, a_ = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                           early_stopping=False).fit(X[tr], y[tr])
        b.append(balanced_accuracy_score(y[te], m.predict(X[te])))
        a_.append(roc_auc_score(y[te] == "solventless",
                                m.predict_proba(X[te])[:, list(m.classes_).index("solventless")]))
    out.append(f"rows={len(D)}  bal_acc {100*np.mean(b):.1f}% (+/-{100*np.std(b):.1f})  AUC {np.mean(a_):.3f}")

    out += ["", "=== 3. producer rows by lab_state (from lab_states.txt laboratories) ==="]
    nv_labs = {"G3 LABS LLC", "NV CANN LABS LLC", "DB LABS LLC", "DPL NV LLC",
               "374 LABS LLC", "ERP LLC", "MA & ASSOCIATES LLC"}
    n_nv = int(P["lab"].isin(nv_labs).sum())
    out.append(f"producer rows from the seven laboratories stamped NV: {n_nv} of {len(P)} "
               f"({100*n_nv/len(P):.1f}%)")
    out.append(f"other laboratories among producer rows: "
               f"{ {k: int(v) for k, v in P.loc[~P['lab'].isin(nv_labs), 'lab'].value_counts().items()} }")

    out += ["", "=== 4. rows with no extractable strain key ==="]
    name = df["product_name"].fillna("").astype(str)
    key = name.map(strain_key)
    nokey = key == ""
    out.append(f"no key: {int(nokey.sum())}   of which have no product name at all: "
               f"{int((nokey & (name.str.strip() == '')).sum())}   "
               f"named rows with no key: {int((nokey & (name.str.strip() != '')).sum())} "
               f"of {int((name.str.strip() != '').sum())} named rows")

    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
