"""Independent audit of the controlled-experiment claims in 3.3.

Recomputes everything from the raw Zenodo xlsx rather than from
phase5_controlled.txt, then asks what that log cannot answer: whether the
single-seed 56.7% is typical of its seed distribution, whether the drop to
38.9% comes from the variety structure or merely from using six folds (a
random-group control keeps the fold count and breaks the variety link),
whether either score is distinguishable from chance by interval or
permutation, and whether the process settings t/T/P, which the paper
excludes, would have leaked the method. Parallelised over seeds.

Output: docs/evidence/audit_phase5_controlled.txt
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/evidence/audit_phase5_controlled.txt"


class _Tee:
    def __init__(self, path):
        self._fh = open(path, "w", encoding="utf-8")
        self._out = sys.stdout

    def write(self, t):
        self._out.write(t)
        self._fh.write(t)

    def flush(self):
        self._out.flush()
        self._fh.flush()


sys.stdout = _Tee(OUT)

SEED = 20260921
FEATS = ["CBD", "CBG", "CBN", "THC"]
d = pd.read_excel(ROOT / "data/raw/zenodo_13823859/BD2 canabinoide  flor hplc.xlsx",
                  sheet_name="Rstudio")
keys = ["Metodo", "Variedad", "t", "T", "P"]
d["R"] = pd.to_numeric(d["R"], errors="coerce")
wide = d.pivot_table(index=keys, columns="C", values="R", aggfunc="mean").reset_index()
wide.columns.name = None
for c in ["Metodo", "Variedad"]:
    wide[c] = wide[c].astype(str).str.strip()
X = wide[FEATS].astype(float).to_numpy()
ym = wide["Metodo"].to_numpy(); yv = wide["Variedad"].to_numpy()
g = wide["Variedad"].to_numpy()


def mk():
    return HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                          early_stopping=False)


def run(X, y, splits):
    bal, f1s = [], []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        p = mk().fit(X[tr], y[tr]).predict(X[te])
        bal.append(balanced_accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return np.array(bal), np.array(f1s)


b5, f5 = run(X, ym, list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, ym)))
bl, fl = run(X, ym, list(LeaveOneGroupOut().split(X, ym, g)))
bv, fv = run(X, yv, list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, yv)))
P = Parallel(n_jobs=-1, verbose=0)

print("== 2b. SEED SWEEP: random CV at k=5 and k=6, 200 seeds each ==")


def seedrun(k, s):
    return run(X, ym, list(StratifiedKFold(k, shuffle=True, random_state=s).split(X, ym)))[0].mean()


for k in (5, 6):
    r = np.array(P(delayed(seedrun)(k, s) for s in range(200)))
    print(f"  k={k}: mean={100*r.mean():.1f}% sd={100*r.std():.1f} "
          f"median={100*np.median(r):.1f} "
          f"2.5-97.5pct=[{100*np.percentile(r,2.5):.1f},{100*np.percentile(r,97.5):.1f}]")
    if k == 5:
        m5 = r
        print(f"  paper's 56.7% sits at the {100*(r < b5.mean()).mean():.0f}th "
              f"percentile of the k=5 seed distribution")
    else:
        m6 = r

print("\n== 2c. CONTROL: 6 RANDOM groups of 27 (fold count + grouping kept, ==")
print("        variety link broken), 200 draws ==")


def ctrlrun(s):
    rng = np.random.default_rng(s)
    fake = np.repeat(np.arange(6), 27); rng.shuffle(fake)
    return run(X, ym, list(LeaveOneGroupOut().split(X, ym, fake)))[0].mean()


ctrl = np.array(P(delayed(ctrlrun)(s) for s in range(200)))
print(f"  mean={100*ctrl.mean():.1f}% sd={100*ctrl.std():.1f} "
      f"2.5-97.5pct=[{100*np.percentile(ctrl,2.5):.1f},{100*np.percentile(ctrl,97.5):.1f}]")
print(f"  real LOVO = {100*bl.mean():.1f}%; "
      f"P(random-group LOVO <= real LOVO) = {(ctrl <= bl.mean()).mean():.3f}")
print("  -> if this is ~0.0x, the variety structure (not the fold count) causes the drop")

print("\n== 3. CONFIDENCE INTERVALS AND TESTS VS CHANCE ==")


def ci(a):
    se = a.std(ddof=1) / np.sqrt(len(a)); t = stats.t.ppf(.975, len(a) - 1)
    return a.mean() - t * se, a.mean() + t * se


for tag, a, ch in [("random 5-fold", b5, 1/3), ("LOVO", bl, 1/3), ("variety", bv, 1/6)]:
    lo, hi = ci(a); t, p = stats.ttest_1samp(a, ch)
    print(f"  {tag:14} {100*a.mean():5.1f}%  folds={np.round(100*a,1).tolist()}")
    print(f"                 95%CI=[{100*lo:.1f},{100*hi:.1f}] contains {100*ch:.1f}%? "
          f"{lo <= ch <= hi}   t={t:.2f} p={p:.4f}")
    print(f"                 paper's mean+/-sd band = [{100*(a.mean()-a.std()):.1f},"
          f"{100*(a.mean()+a.std()):.1f}] contains chance? "
          f"{(a.mean()-a.std()) <= ch <= (a.mean()+a.std())}")

print("\n== 3b. POOLED-PREDICTION BINOMIAL CI ==")
for tag, splits, y, nc in [
        ("method random5", list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, ym)), ym, 3),
        ("method LOVO", list(LeaveOneGroupOut().split(X, ym, g)), ym, 3),
        ("variety random5", list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, yv)), yv, 6)]:
    pr = np.empty(len(y), dtype=object)
    for tr, te in splits:
        pr[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    pr = np.array(list(pr))
    k = int((pr == y).sum()); n = len(y)
    lo, hi = stats.binomtest(k, n).proportion_ci(.95, method="wilson")
    pv = stats.binomtest(k, n, 1/nc, alternative="greater").pvalue
    print(f"  {tag:16} {k}/{n} plain acc={100*k/n:.1f}% CI=[{100*lo:.1f},{100*hi:.1f}] "
          f"pooled bal_acc={100*balanced_accuracy_score(y,pr):.1f}%  "
          f"p(>chance {100/nc:.1f}%)={pv:.2e} CI_contains_chance={lo <= 1/nc <= hi}")

print("\n== 3c. PERMUTATION TESTS (300 reps) ==")


def perm_method(s, scheme):
    rr = np.random.default_rng(s); yp = rr.permutation(ym)
    sp = (list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, yp))
          if scheme == "r5" else list(LeaveOneGroupOut().split(X, yp, g)))
    return run(X, yp, sp)[0].mean()


for scheme, obs in [("r5", b5.mean()), ("lovo", bl.mean())]:
    null = np.array(P(delayed(perm_method)(s, scheme) for s in range(300)))
    print(f"  {scheme:5} obs={100*obs:.1f}%  null mean={100*null.mean():.1f}% "
          f"sd={100*null.std():.1f} 95pct={100*np.percentile(null,95):.1f}%  "
          f"p={((null>=obs).sum()+1)/(len(null)+1):.4f}")


def perm_var(s):
    rr = np.random.default_rng(1000+s); yp = rr.permutation(yv)
    return run(X, yp, list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, yp)))[0].mean()


nv = np.array(P(delayed(perm_var)(s) for s in range(300)))
print(f"  variety obs={100*bv.mean():.1f}% null mean={100*nv.mean():.1f}% "
      f"95pct={100*np.percentile(nv,95):.1f}% p={((nv>=bv.mean()).sum()+1)/(len(nv)+1):.4f}")

print("\n== 3d. IS 56.7% vs 38.9% DISTINGUISHABLE? ==")
tt = stats.ttest_ind(b5, bl, equal_var=False)
u = stats.mannwhitneyu(b5, bl, alternative="greater")
print(f"  gap = {100*(b5.mean()-bl.mean()):.1f} pts (paper says 18)")
print(f"  Welch t={tt.statistic:.2f} p={tt.pvalue:.4f}  Mann-Whitney p={u.pvalue:.4f}")
print(f"  honest gap vs k=6 random ({100*m6.mean():.1f}%): "
      f"{100*(m6.mean()-bl.mean()):.1f} pts")
print(f"  P(random 5-fold seed <= LOVO 38.9%) = {(m5 <= bl.mean()).mean():.3f}")

print("\n== 4. VARIETY TASK ==")
print("  class counts:", wide["Variedad"].value_counts().to_dict())
print(f"  60.7/16.667 = {bv.mean()/(1/6):.2f}x chance; paper: 'nearly four times'")

print("\n== 5/6. PREPROCESSING & LEAK CHECKS ==")
print("  duplicate (keys,C) cells needing aggfunc=mean:",
      int((d.groupby(keys+['C']).size() > 1).sum()))
print("  raw Variedad labels:", sorted(d["Variedad"].unique()))
print("  P ranges per method (perfect method separator, NOT in X):")
print(wide.groupby("Metodo")["P"].agg(["min", "max", "nunique"]).to_string())
X2 = wide[FEATS+["t", "T", "P"]].astype(float).to_numpy()
b, f = run(X2, ym, list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X2, ym)))
print(f"  if t/T/P were included: bal={100*b.mean():.1f}% (paper correctly excludes them)")
