"""Phase 4: does the chemistry generalise, or is it riding producer/lab?

Phase 3 raised a real alarm — the profile predicts which lab ran the assay at
55.6% against a 40.3% baseline (+15.3 points). A naive random split also gives
79.6% balanced accuracy on the actual task, but that number is optimistic by
construction: the same producer, and the same lab, sit on both sides of the
split.

So the task is scored under splits that remove each shortcut in turn:

  random          optimistic reference, not a result
  unseen-producer GroupKFold by producer — cannot memorise the maker
  unseen-lab      leave-one-lab-out — cannot memorise the instrument
  dual-class-only trained and tested only on the 55 producers that make BOTH
                  classes, where producer identity carries no class information

The last one is the strictest available evidence: inside it, a model that
knows only "who made this" is at chance, so anything above baseline has to
come from the chemistry.

Metrics are balanced accuracy and macro F1, never plain accuracy — the classes
are 2:1 and accuracy would reward always guessing hydrocarbon.

Output: docs/evidence/phase4_generalisation.txt
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data/labeled/master.csv"
OUT = ROOT / "docs/evidence/phase4_generalisation.txt"
SEED = 20260921


def model():
    return HistGradientBoostingClassifier(
        max_iter=200, random_state=SEED, early_stopping=False)


def features(df: pd.DataFrame) -> list[str]:
    flags = [c for c in df.columns if c.endswith("_tested")]
    cols = [c[: -len("_tested")] for c in flags]
    return sorted(c for c in cols
                  if c in df.columns and df[c].notna().sum() > 0)


def evaluate(X, y, splits, name: str, out: list[str]) -> float:
    """Run one split scheme; report balanced accuracy and macro F1."""
    bal, f1s, recalls = [], [], []
    cms = np.zeros((2, 2), dtype=int)
    labels = ["hydrocarbon", "solventless"]

    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = model().fit(X[tr], y[tr])
        p = clf.predict(X[te])
        bal.append(balanced_accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
        recalls.append(recall_score(y[te], p, average=None, labels=labels))
        cms += confusion_matrix(y[te], p, labels=labels)

    if not bal:
        out.append(f"{name}: no usable folds")
        return 0.0

    r = np.mean(recalls, axis=0)
    out.append(f"{name:18} folds={len(bal):>2}  "
               f"balanced_acc={100*np.mean(bal):>5.1f}% (+/-{100*np.std(bal):.1f})  "
               f"macro_F1={100*np.mean(f1s):>5.1f}%  "
               f"recall[hydro={100*r[0]:.1f}% solventless={100*r[1]:.1f}%]")
    out.append(f"{'':18} confusion (rows=true hydro/solventless): "
               f"{cms.tolist()}")
    return float(np.mean(bal))


def main() -> int:
    df = pd.read_csv(MASTER, low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    feats = features(df)

    out: list[str] = []
    out.append(f"rows: {len(df)}   features: {len(feats)}")
    out.append(f"class balance: "
               f"{df['label'].value_counts().to_dict()}")
    out.append("")

    X = df[feats].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()

    # --- baselines -------------------------------------------------------
    out.append("=== BASELINES (what 'learning nothing' looks like) ===")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for strat in ["most_frequent", "stratified"]:
        b = []
        for tr, te in cv.split(X, y):
            d = DummyClassifier(strategy=strat, random_state=SEED)
            d.fit(X[tr], y[tr])
            b.append(balanced_accuracy_score(y[te], d.predict(X[te])))
        out.append(f"{strat:18} balanced_acc={100*np.mean(b):>5.1f}%")
    out.append("")

    # --- split schemes ---------------------------------------------------
    out.append("=== THE TASK UNDER INCREASINGLY HONEST SPLITS ===")
    evaluate(X, y, list(cv.split(X, y)), "random (optimistic)", out)

    # fillna before strip: missing producers arrive as float NaN, which would
    # poison any comparison or sort against the string values.
    prod = df["producer"].fillna("").astype(str).str.strip()
    has_prod = prod != ""
    if has_prod.sum() > 1000 and prod[has_prod].nunique() >= 5:
        Xp, yp = X[has_prod.to_numpy()], y[has_prod.to_numpy()]
        gp = prod[has_prod].to_numpy()
        n = min(5, len(np.unique(gp)))
        evaluate(Xp, yp, list(GroupKFold(n_splits=n).split(Xp, yp, gp)),
                 "unseen-producer", out)

    lab = df["lab"].fillna("").astype(str).str.strip()
    labs = lab.value_counts()
    big = [l for l, c in labs.items() if c >= 500 and l]
    folds = []
    for held in big:
        te = (lab == held).to_numpy()
        tr = ~te
        if y[te].size and len(np.unique(y[te])) == 2:
            folds.append((np.where(tr)[0], np.where(te)[0]))
    if folds:
        evaluate(X, y, folds, "unseen-lab", out)

    # --- the strictest test ---------------------------------------------
    out.append("")
    out.append("=== DUAL-CLASS PRODUCERS ONLY ===")
    out.append("producers that make BOTH classes: knowing the maker tells you")
    out.append("nothing about the label, so any lift is chemistry")
    per = df[has_prod].groupby(prod[has_prod])["label"].nunique()
    dual = set(per[per == 2].index)
    m = has_prod & prod.isin(dual)
    out.append(f"producers: {len(dual)}   rows: {int(m.sum())}")
    if m.sum() > 1000:
        Xd, yd = X[m.to_numpy()], y[m.to_numpy()]
        gd = prod[m].to_numpy()
        out.append(f"class balance: {pd.Series(yd).value_counts().to_dict()}")
        maj = pd.Series(yd).value_counts().max() / len(yd)
        out.append(f"(plain-accuracy majority here would be {100*maj:.1f}%, "
                   f"balanced-accuracy baseline is 50.0%)")
        n = min(5, len(np.unique(gd)))
        evaluate(Xd, yd, list(GroupKFold(n_splits=n).split(Xd, yd, gd)),
                 "dual-class+unseen", out)

    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
