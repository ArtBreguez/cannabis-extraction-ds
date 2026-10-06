"""Pass twelve: checks a fourth referee asked for, all on the producer rows.

  A. the temporal split restricted to rows that name a producer (the pass-ten
     run included the producer-less rows, which score near 100% on their
     reporting panel and inflated it)
  B. the stable period: random against producer-held-out on 2020-2022 only,
     before the 2023 change in how ethanol extracts are labelled
  D. duplicates among the producer rows, and the headline pair without them
  E. rows whose only reported analyte is total_terpenes = 0
  F. producer names that are spellings of one company
  G. per-class recall of the random split on the producer rows
  H. name-path agreement as recorded in master.csv
  I. AUC over the twenty partitions for the reported (unweighted) model
  K-M. small counts the text relies on

Output: docs/evidence/review12_checks.txt
"""
from __future__ import annotations

import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.phase4_generalisation import SEED, features  # noqa: E402

OUT = ROOT / "docs/evidence/review12_checks.txt"
POS = "solventless"


def hgb():
    return HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                          early_stopping=False)


def ev(X, y, splits):
    b, a, r = [], [], []
    for tr, te in splits:
        m = hgb().fit(X[tr], y[tr])
        p = m.predict(X[te])
        s = m.predict_proba(X[te])[:, list(m.classes_).index(POS)]
        b.append(balanced_accuracy_score(y[te], p))
        a.append(roc_auc_score(y[te] == POS, s))
        r.append(recall_score(y[te], p, average=None, labels=["hydrocarbon", POS]))
    return np.array(b), np.array(a), np.mean(r, axis=0)


def f(a):
    return f"{100*a.mean():.1f}% (+/-{100*a.std():.1f})"


def main() -> int:
    df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    feats = features(df)
    prod = df["producer"].fillna("").astype(str).str.strip()
    has = (prod != "").to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = [f"rows: {len(df)}   features: {len(feats)}"]

    def say(line=""):
        out.append(line)
        print(line, flush=True)

    P = df[has].copy()
    X = P[feats].astype(float).to_numpy()
    y = P["label"].astype(str).to_numpy()
    g = prod[has].to_numpy()
    dt = pd.to_datetime(P["date_tested"], errors="coerce")

    def temporal(mask, label):
        idx = np.where(mask & dt.notna().to_numpy())[0]
        idx = idx[np.argsort(dt.iloc[idx].to_numpy(), kind="stable")]
        cut = int(0.8 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        m = hgb().fit(X[tr], y[tr])
        p = m.predict(X[te])
        s = m.predict_proba(X[te])[:, list(m.classes_).index(POS)]
        seen = np.mean(np.isin(g[te], np.unique(g[tr])))
        say(f"  {label}: dated producer rows={len(idx)}  train up to {str(dt.iloc[tr[-1]])[:10]}  "
            f"test rows={len(te)}  bal_acc {100*balanced_accuracy_score(y[te], p):.1f}%  "
            f"AUC {roc_auc_score(y[te] == POS, s):.3f}  "
            f"test rows from producers seen in training: {100*seen:.1f}%")

    say()
    say("=== A. temporal split, producer rows only ===")
    temporal(np.ones(len(P), dtype=bool), "all years")
    nop = df[~has]
    dn = pd.to_datetime(nop["date_tested"], errors="coerce")
    say(f"  producer-less rows with a date: {int(dn.notna().sum())} of {len(nop)}, "
        f"from {str(dn.min())[:10]} to {str(dn.max())[:10]}")

    say()
    say("=== B. the stable period: producer rows tested in 2020-2022 ===")
    yr = dt.dt.year.to_numpy()
    early = np.isin(yr, [2020, 2021, 2022])
    late = np.isin(yr, [2023, 2024])
    for label, m in (("2020-2022", early), ("2023-2024", late)):
        rb, ra, _ = ev(X[m], y[m], list(cv.split(X[m], y[m])))
        gb, ga, _ = ev(X[m], y[m], list(GroupKFold(5).split(X[m], y[m], g[m])))
        say(f"  {label}: rows={int(m.sum())} producers={len(set(g[m]))}  "
            f"random {f(rb)} AUC {ra.mean():.3f}   held out {f(gb)} AUC {ga.mean():.3f}   "
            f"gap {100*(rb.mean()-gb.mean()):.1f} points")
    temporal(early, "2020-2022 only")

    say()
    say("=== D. duplicates on product_name + lab + producer + 19 analytes ===")
    key = ["product_name", "lab", "producer"] + feats
    dup_all = df.duplicated(subset=key).to_numpy()
    say(f"  all usable rows: {int(dup_all.sum())}   of which name a producer: {int((dup_all & has).sum())}   "
        f"producer-less: {int((dup_all & ~has).sum())}")
    dp = P.duplicated(subset=key).to_numpy()
    rb, ra, _ = ev(X[~dp], y[~dp], list(cv.split(X[~dp], y[~dp])))
    gb, ga, _ = ev(X[~dp], y[~dp], list(GroupKFold(5).split(X[~dp], y[~dp], g[~dp])))
    say(f"  producer rows without duplicates: rows={int((~dp).sum())}  random {f(rb)}   "
        f"held out {f(gb)}   gap {100*(rb.mean()-gb.mean()):.1f} points")

    say()
    say("=== E. producer rows whose only reported analyte is total_terpenes = 0 ===")
    npres = P[feats].notna().sum(axis=1).to_numpy()
    only = (npres == 1) & (P["total_terpenes"].fillna(-1).to_numpy() == 0)
    say(f"  rows: {int(only.sum())}  classes: { {k: int(v) for k, v in Counter(y[only]).items()} }  "
        f"of the duplicates above: {int((only & dp).sum())}")
    k = ~only
    rb, ra, _ = ev(X[k], y[k], list(cv.split(X[k], y[k])))
    gb, ga, _ = ev(X[k], y[k], list(GroupKFold(5).split(X[k], y[k], g[k])))
    say(f"  without them: rows={int(k.sum())}  random {f(rb)} AUC {ra.mean():.3f}   "
        f"held out {f(gb)} AUC {ga.mean():.3f}   gap {100*(rb.mean()-gb.mean()):.1f} points")

    say()
    say("=== F. producer names that are spellings of one company ===")
    norm = np.array([re.sub(r"[^A-Z0-9]", "", s.upper()) for s in g])
    by = {}
    for raw, n_ in zip(g, norm):
        by.setdefault(n_, set()).add(raw)
    multi = {k_: v for k_, v in by.items() if len(v) > 1}
    say(f"  distinct producer strings: {len(set(g))}   after normalising case and punctuation: {len(set(norm))}")
    for k_, v in multi.items():
        say(f"  merged: {sorted(v)}  rows={int(np.isin(g, list(v)).sum())}")
    gb, ga, _ = ev(X, y, list(GroupKFold(5).split(X, y, norm)))
    say(f"  held out with normalised producer groups: {f(gb)} AUC {ga.mean():.3f}")

    say()
    say("=== G. random split on the producer rows: per-class recall ===")
    rb, ra, rr = ev(X, y, list(cv.split(X, y)))
    say(f"  bal_acc {f(rb)}  AUC {ra.mean():.3f}  recall hydro/solventless {100*rr[0]:.1f}/{100*rr[1]:.1f}%")

    say()
    say("=== H. name-path agreement as recorded in master.csv ===")
    full = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    named = full["name_label"].fillna("").astype(str).str.strip() != ""
    conf = int((full["label_conflict"] != 0).sum())
    say(f"  labelled rows: {len(full)}   with a name-path label: {int(named.sum())} "
        f"({100*named.mean():.1f}%)   agree: {int(named.sum()) - conf}   conflict: {conf}   "
        f"agreement {100*(int(named.sum()) - conf)/int(named.sum()):.1f}%")

    say()
    say("=== I. the reported model over twenty shuffled partitions: AUC ===")
    ba, au = [], []
    for s in range(20):
        gb, ga, _ = ev(X, y, list(GroupKFold(5, shuffle=True, random_state=s).split(X, y, g)))
        ba.append(gb.mean()); au.append(ga.mean())
    ba, au = np.array(ba), np.array(au)
    say(f"  bal_acc min {100*ba.min():.1f}%  median {100*np.median(ba):.1f}%  max {100*ba.max():.1f}%   "
        f"AUC min {au.min():.3f}  median {np.median(au):.3f}  max {au.max():.3f}")

    say()
    say("=== K-M. small counts ===")
    pre = df["product_type"].fillna("").str.lower().str.contains("pre-roll").to_numpy()
    say(f"  pre-roll rows: {int(pre.sum())}   of which name a producer: {int((pre & has).sum())}")
    nm = P["product_name"].fillna("").str.lower()
    eth = nm.str.contains("ethanol|etoh").to_numpy()
    sol = (y == POS)
    say(f"  ethanol-named non-solvent rows: {int((eth & sol).sum())}  with a date: {int((eth & sol & dt.notna().to_numpy()).sum())}   "
        f"ethanol-named solvent-based rows: {int((eth & ~sol).sum())}  with a date: {int((eth & ~sol & dt.notna().to_numpy()).sum())}")
    mech = nm.str.contains(r"rosin|hash|kief|bubble|ice\s*water|solventless").to_numpy()
    hydw = nm.str.contains(r"\bbho(?![a-z])|butane|hydrocarbon|live\s*resin|shatter|badder|batter|\bwax\b").to_numpy()
    say(f"  mechanical-word non-solvent rows: {int((mech & sol).sum())}   hydrocarbon-word solvent-based rows: {int((hydw & ~sol).sum())}   "
        f"of those also hitting a mechanical word (excluded from the named subset): {int((hydw & ~sol & mech).sum())}")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
