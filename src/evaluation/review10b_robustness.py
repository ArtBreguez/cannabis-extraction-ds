"""Pass ten, part B: robustness checks a second referee asked for.

  C. a second, low-variance estimator (standardised logistic regression) on
     both datasets, with a permutation test and a seed sweep on the
     controlled data
  D. the gap against model capacity (max_iter)
  E. label-definition subsets: non-distillate rows, and the rows whose
     product names say rosin/hash against those that say BHO/live resin
  F. a temporal split
  G. the binary task at learning_rate=0.05
  H. same-sample random splits for the two strain-grouped subsets of 4.3

Output: docs/evidence/review10b_robustness.txt
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.phase4_generalisation import SEED, features  # noqa: E402
from src.models.phase4b_strain import strain_key  # noqa: E402
from src.models.phase5_controlled import load_wide  # noqa: E402

OUT = ROOT / "docs/evidence/review10b_robustness.txt"


def hgb(max_iter=200, lr=0.1):
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                          random_state=SEED, early_stopping=False)


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def linear_matrix(df, feats):
    """Linear models need finite inputs: log1p of the value clipped at zero
    (which also tames the unit-artefact potencies of 2.2), empty cells set to
    zero, plus one 0/1 column per analyte saying whether it was reported."""
    v = df[feats].astype(float)
    return np.hstack([np.log1p(v.clip(lower=0).fillna(0).to_numpy()),
                      v.notna().to_numpy().astype(float)])


def run(make, X, y, splits):
    b = []
    for tr, te in splits:
        b.append(balanced_accuracy_score(y[te], make().fit(X[tr], y[tr]).predict(X[te])))
    return np.array(b)


def fmt(a):
    return f"{100*a.mean():.1f}% (+/-{100*a.std():.1f})"


def main() -> int:
    df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    feats = features(df)
    X = df[feats].astype(float).to_numpy()
    XL = linear_matrix(df, feats)
    y = df["label"].astype(str).to_numpy()
    prod = df["producer"].fillna("").astype(str).str.strip()
    has = (prod != "").to_numpy()
    g = prod.to_numpy()
    per = df[has].groupby(prod[has])["label"].nunique()
    dual = (has & prod.isin(set(per[per == 2].index)).to_numpy())
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = [f"rows: {len(df)}   features: {len(feats)}"]

    def say(line):
        out.append(line)
        print(line, flush=True)

    # C. market data, logistic regression ------------------------------
    say("")
    say("=== C1. market data, standardised logistic regression ===")
    r_all = run(logit, XL, y, list(cv.split(XL, y)))
    r_sub = run(logit, XL[has], y[has], list(cv.split(XL[has], y[has])))
    g_sub = run(logit, XL[has], y[has], list(GroupKFold(5).split(XL[has], y[has], g[has])))
    r_du = run(logit, XL[dual], y[dual], list(cv.split(XL[dual], y[dual])))
    g_du = run(logit, XL[dual], y[dual], list(GroupKFold(5).split(XL[dual], y[dual], g[dual])))
    say(f"  random, all rows            {fmt(r_all)}")
    say(f"  random, producer rows       {fmt(r_sub)}")
    say(f"  unseen-producer             {fmt(g_sub)}   same-sample gap {100*(r_sub.mean()-g_sub.mean()):.1f} points")
    say(f"  dual-class, random          {fmt(r_du)}")
    say(f"  dual-class, unseen-producer {fmt(g_du)}   gap {100*(r_du.mean()-g_du.mean()):.1f} points")

    # C. controlled data -------------------------------------------------
    say("")
    say("=== C2. controlled data, two estimators ===")
    w = load_wide()
    F = ["CBD", "CBG", "CBN", "THC"]
    Xc = w[F].astype(float).to_numpy()
    ym = w["Metodo"].astype(str).to_numpy()
    yv = w["Variedad"].astype(str).to_numpy()
    cvc = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, make in (("gradient boosting", hgb), ("logistic regression", logit)):
        r = run(make, Xc, ym, list(cvc.split(Xc, ym)))
        lo = run(make, Xc, ym, list(LeaveOneGroupOut().split(Xc, ym, yv)))
        va = run(make, Xc, yv, list(cvc.split(Xc, yv)))
        say(f"  {name:20} method random {fmt(r)}   method LOVO {fmt(lo)}   "
            f"variety random {fmt(va)}   gap {100*(r.mean()-lo.mean()):.1f} points")
    lo_obs = run(logit, Xc, ym, list(LeaveOneGroupOut().split(Xc, ym, yv))).mean()
    rng = np.random.default_rng(1)
    null = np.array([run(logit, Xc, yp, list(LeaveOneGroupOut().split(Xc, yp, yv))).mean()
                     for yp in (rng.permutation(ym) for _ in range(1000))])
    k = int((null >= lo_obs).sum())
    say(f"  logistic LOVO permutation: obs {100*lo_obs:.1f}%  null mean {100*null.mean():.1f}% "
        f"sd {100*null.std():.1f}  shuffles at or above obs: {k} of 1000  p={(k+1)/1001:.3f}")
    sweep = np.array([run(logit, Xc, ym, list(StratifiedKFold(5, shuffle=True, random_state=s)
                                              .split(Xc, ym))).mean() for s in range(200)])
    say(f"  logistic random 5-fold over 200 seeds: mean {100*sweep.mean():.1f}% sd {100*sweep.std():.1f}"
        f"   seed-averaged gap to LOVO {100*(sweep.mean()-lo_obs):.1f} points")

    # D. capacity --------------------------------------------------------
    say("")
    say("=== D. gap against capacity (producer rows, gradient boosting) ===")
    for mi in (50, 100, 200, 400):
        r = run(lambda: hgb(mi), X[has], y[has], list(cv.split(X[has], y[has])))
        gg = run(lambda: hgb(mi), X[has], y[has], list(GroupKFold(5).split(X[has], y[has], g[has])))
        say(f"  max_iter={mi:<4} random {fmt(r)}   unseen-producer {fmt(gg)}   gap {100*(r.mean()-gg.mean()):.1f} points")

    # E. label-definition subsets ---------------------------------------
    say("")
    say("=== E. label-definition subsets (gradient boosting) ===")
    name = df["product_name"].fillna("").str.lower()
    dist = name.str.contains("distillate|disty").to_numpy()
    mech = name.str.contains(r"rosin|hash|kief|bubble|ice\s*water|solventless").to_numpy()
    hydw = name.str.contains(r"\bbho(?![a-z])|butane|hydrocarbon|live\s*resin|shatter|badder|batter|\bwax\b").to_numpy()
    for label, m in (("non-distillate rows", ~dist),
                     ("named rows: mechanical-word solventless vs hydrocarbon-word hydrocarbon",
                      (mech & (y == "solventless")) | (hydw & (y == "hydrocarbon") & ~mech))):
        mh = m & has
        say(f"  {label}: rows={int(m.sum())} with producer={int(mh.sum())} "
            f"producers={len(set(g[mh]))} classes={ {k: int(v) for k, v in pd.Series(y[mh]).value_counts().items()} }")
        r = run(hgb, X[mh], y[mh], list(cv.split(X[mh], y[mh])))
        gg = run(hgb, X[mh], y[mh], list(GroupKFold(5).split(X[mh], y[mh], g[mh])))
        say(f"    random (producer rows) {fmt(r)}   unseen-producer {fmt(gg)}   gap {100*(r.mean()-gg.mean()):.1f} points")

    # F. temporal split --------------------------------------------------
    say("")
    say("=== F. temporal split (train on the earliest 80% of dated rows) ===")
    dt = pd.to_datetime(df["date_tested"], errors="coerce")
    ok = dt.notna().to_numpy()
    order = np.argsort(dt[ok].to_numpy())
    idx = np.where(ok)[0][order]
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    p = hgb().fit(X[tr], y[tr]).predict(X[te])
    say(f"  dated rows={len(idx)}  train up to {str(dt.iloc[tr[-1]])[:10]}  test rows={len(te)}  "
        f"balanced_acc {100*balanced_accuracy_score(y[te], p):.1f}%  "
        f"test producers seen in training: {100*np.mean([gg in set(g[tr]) for gg in g[te]]):.1f}% of test rows")

    # G. learning rate ---------------------------------------------------
    say("")
    say("=== G. the binary task at learning_rate=0.05 ===")
    r = run(lambda: hgb(200, 0.05), X, y, list(cv.split(X, y)))
    gg = run(lambda: hgb(200, 0.05), X[has], y[has], list(GroupKFold(5).split(X[has], y[has], g[has])))
    say(f"  random, all rows {fmt(r)}   unseen-producer {fmt(gg)}")

    # H. strain subsets --------------------------------------------------
    say("")
    say("=== H. same-sample random splits for the strain-grouped subsets of 4.3 ===")
    key = df["product_name"].fillna("").map(strain_key)
    hk = (key != "").to_numpy()
    r = run(hgb, X[hk], y[hk], list(cv.split(X[hk], y[hk])))
    say(f"  rows with a strain key={int(hk.sum())}  random {fmt(r)}")
    pk = pd.DataFrame({"k": key[hk].to_numpy(), "y": y[hk]}).groupby("k")["y"].nunique()
    dk = hk & key.isin(set(pk[pk == 2].index)).to_numpy()
    r = run(hgb, X[dk], y[dk], list(cv.split(X[dk], y[dk])))
    say(f"  dual-class strain rows={int(dk.sum())}  random {fmt(r)}")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
