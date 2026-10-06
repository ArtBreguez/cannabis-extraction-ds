"""Pass nine: the checks an independent referee asked for on the market data.

  B. the 2.3 missingness probe in balanced accuracy, and the main model's
     plain accuracy, so 2.3 and 2.5 compare like with like
  C. a same-sample random split on the dual-class rows, so the dual-class
     grouped score has its own like-with-like comparator
  D. reseeding that actually moves something: GroupKFold with shuffled group
     assignment, since the unshuffled grouped run is deterministic
  E. duplicate rows: how many, and whether dropping them moves the headline
  H. per-fold macro F1 and the two macro-F1 baselines
  K. the pure missingness patterns inside each producer-less laboratory
  P. pooled balanced accuracy of the unseen-producer confusion matrix
  Z. the controlled data: are the 162 samples distinct design cells?

Output: docs/evidence/review9_checks.txt
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.phase4_generalisation import SEED, features  # noqa: E402

OUT = ROOT / "docs/evidence/review9_checks.txt"


def clf():
    return HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                          early_stopping=False)


def run(X, y, splits):
    bal, f1s, acc = [], [], []
    yt, yp = [], []
    for tr, te in splits:
        p = clf().fit(X[tr], y[tr]).predict(X[te])
        bal.append(balanced_accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
        acc.append((p == y[te]).mean())
        yt.append(y[te]); yp.append(p)
    return (np.array(bal), np.array(f1s), np.array(acc),
            np.concatenate(yt), np.concatenate(yp))


def fmt(a):
    return f"{100*a.mean():.1f}% (+/-{100*a.std():.1f})"


def main() -> int:
    df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    feats = features(df)
    out = [f"rows: {len(df)}   features: {len(feats)}"]
    X = df[feats].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()
    prod = df["producer"].fillna("").astype(str).str.strip()
    has = (prod != "").to_numpy()
    cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # B --------------------------------------------------------------
    out += ["", "=== B. the 2.3 missingness probe, both metrics ==="]
    pat = [tuple(int(v) for v in r) for r in df[feats].notna().to_numpy()]
    hits, yt, yp = 0, [], []
    for tr, te in cv5.split(X, y):
        vote = defaultdict(Counter)
        for i in tr:
            vote[pat[i]][y[i]] += 1
        maj = Counter(y[tr]).most_common(1)[0][0]
        pred = np.array([vote[pat[i]].most_common(1)[0][0] if pat[i] in vote else maj
                         for i in te])
        yt.append(y[te]); yp.append(pred)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    out.append(f"  missingness-only probe: plain acc={100*(yt==yp).mean():.1f}%  "
               f"balanced_acc={100*balanced_accuracy_score(yt, yp):.1f}%  "
               f"(majority baseline plain 65.8%, balanced 50.0%)")
    b, f, a, yt, yp = run(X, y, list(cv5.split(X, y)))
    out.append(f"  main model, random split: plain acc={fmt(a)}  balanced_acc={fmt(b)}")

    # C --------------------------------------------------------------
    out += ["", "=== C. dual-class rows: random split on the SAME rows ==="]
    per = df[has].groupby(prod[has])["label"].nunique()
    dual = set(per[per == 2].index)
    md = (has & prod.isin(dual)).to_numpy()
    Xd, yd, gd = X[md], y[md], prod[md].to_numpy()
    bd_r, fd_r, _, _, _ = run(Xd, yd, list(cv5.split(Xd, yd)))
    bd_g, fd_g, _, _, _ = run(Xd, yd, list(GroupKFold(5).split(Xd, yd, gd)))
    out.append(f"  rows={len(yd)}  random 5-fold {fmt(bd_r)}   "
               f"grouped by producer {fmt(bd_g)}   gap {100*(bd_r.mean()-bd_g.mean()):.1f} points")

    # D --------------------------------------------------------------
    out += ["", "=== D. grouped schemes with SHUFFLED group-to-fold assignment ==="]
    Xp, yp_, gp = X[has], y[has], prod[has].to_numpy()
    for s in (1, 7, 12345):
        bu, _, _, _, _ = run(Xp, yp_, list(GroupKFold(5, shuffle=True, random_state=s)
                                           .split(Xp, yp_, gp)))
        bdd, _, _, _, _ = run(Xd, yd, list(GroupKFold(5, shuffle=True, random_state=s)
                                           .split(Xd, yd, gd)))
        out.append(f"  seed {s:>6}: unseen-producer {fmt(bu)}   dual-class {fmt(bdd)}")

    # E --------------------------------------------------------------
    out += ["", "=== E. duplicate rows ==="]
    key = ["product_name", "lab", "producer"] + feats
    dup_full = int(df.duplicated().sum())
    dup_key = df.duplicated(subset=key)
    out.append(f"  exact duplicate rows (all columns): {dup_full}")
    out.append(f"  duplicates on product_name+lab+producer+19 analytes: {int(dup_key.sum())} "
               f"({100*dup_key.mean():.1f}% of usable rows)")
    dd = df[~dup_key]
    Xe = dd[feats].astype(float).to_numpy(); ye = dd["label"].astype(str).to_numpy()
    pe = dd["producer"].fillna("").astype(str).str.strip(); he = (pe != "").to_numpy()
    be_r, _, _, _, _ = run(Xe, ye, list(cv5.split(Xe, ye)))
    be_g, _, _, _, _ = run(Xe[he], ye[he], list(GroupKFold(5).split(Xe[he], ye[he], pe[he].to_numpy())))
    out.append(f"  after dropping them: rows={len(dd)}  random {fmt(be_r)}  "
               f"unseen-producer {fmt(be_g)}  gap {100*(be_r.mean()-be_g.mean()):.1f} points")

    # H and P --------------------------------------------------------
    out += ["", "=== H/P. unseen-producer and dual-class: macro F1 per fold, pooled matrix ==="]
    bu, fu, _, yt, yp = run(Xp, yp_, list(GroupKFold(5).split(Xp, yp_, gp)))
    out.append(f"  unseen-producer: macro_F1 folds={[round(100*float(v),1) for v in fu]} "
               f"mean={100*fu.mean():.1f}% (+/-{100*fu.std():.1f})")
    out.append(f"  unseen-producer: pooled balanced_acc={100*balanced_accuracy_score(yt, yp):.1f}%  "
               f"pooled macro_F1={100*f1_score(yt, yp, average='macro'):.1f}%")
    out.append(f"  dual-class:      macro_F1 folds={[round(100*float(v),1) for v in fd_g]} "
               f"mean={100*fd_g.mean():.1f}% (+/-{100*fd_g.std():.1f})")
    maj = np.full(len(yp_), "hydrocarbon")
    strat = np.random.default_rng(SEED).choice(yp_, size=len(yp_), replace=False)
    out.append(f"  macro_F1 baselines on the producer rows: majority {100*f1_score(yp_, maj, average='macro'):.1f}%  "
               f"stratified shuffle {100*f1_score(yp_, strat, average='macro'):.1f}%")

    # K --------------------------------------------------------------
    out += ["", "=== K. pure missingness patterns INSIDE each producer-less laboratory ==="]
    for lab in sorted(df.loc[~has, "lab"].unique()):
        sub = df[(df["lab"] == lab)]
        c = Counter()
        for i in np.where((df["lab"] == lab).to_numpy())[0]:
            c[(sum(pat[i]), y[i])] += 1
        out.append(f"  {lab:30} {dict(sorted(c.items()))}")
    out.append("  (key = analytes reported, class)")

    # Z --------------------------------------------------------------
    out += ["", "=== Z. controlled data: distinct design cells ==="]
    d = pd.read_excel(ROOT / "data/raw/zenodo_13823859/BD2 canabinoide  flor hplc.xlsx",
                      sheet_name="Rstudio")
    keys = ["Metodo", "Variedad", "t", "T", "P"]
    cells = d[keys].drop_duplicates()
    out.append(f"  long rows={len(d)}  distinct (method, variety, t, T, P) cells={len(cells)}")

    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
