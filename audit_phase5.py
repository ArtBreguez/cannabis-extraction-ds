"""Independent audit of PREPRINT.md claims about Zenodo 13823859 (dataset 2).
Recomputes everything from the raw xlsx. Does not read author logs.
"""
import numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import (StratifiedKFold, LeaveOneGroupOut,
                                     GroupKFold, KFold)
from scipy import stats
warnings.filterwarnings("ignore")

SEED = 20260921
SRC = Path("data/raw/zenodo_13823859/BD2 canabinoide  flor hplc.xlsx")
FEATS = ["CBD", "CBG", "CBN", "THC"]

d = pd.read_excel(SRC, sheet_name="Rstudio")
keys = ["Metodo", "Variedad", "t", "T", "P"]
d["R"] = pd.to_numeric(d["R"], errors="coerce")
wide = d.pivot_table(index=keys, columns="C", values="R",
                     aggfunc="mean").reset_index()
wide.columns.name = None
for c in ["Metodo", "Variedad"]:
    wide[c] = wide[c].astype(str).str.strip()

X = wide[FEATS].astype(float).to_numpy()
ym = wide["Metodo"].to_numpy()
yv = wide["Variedad"].to_numpy()
g = wide["Variedad"].to_numpy()


def mk():
    return HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                          early_stopping=False)


def run(X, y, splits):
    bal, f1s, ntr, nte = [], [], [], []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        p = mk().fit(X[tr], y[tr]).predict(X[te])
        bal.append(balanced_accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
        ntr.append(len(tr)); nte.append(len(te))
    return np.array(bal), np.array(f1s), ntr, nte


def rep(tag, bal, f1s, ntr):
    print(f"{tag:34} folds={len(bal):>2} bal={100*bal.mean():5.1f}% "
          f"(+/-{100*bal.std():.1f}) macroF1={100*f1s.mean():5.1f}% "
          f"train_n={int(np.mean(ntr))} folds_raw={np.round(100*bal,1).tolist()}")


print("== 0. DESIGN AUDIT (raw file) ==")
print("raw long rows:", len(d), " cannabinoids:", d["C"].value_counts().to_dict())
print("pivoted samples:", len(wide))
print("per method:", wide["Metodo"].value_counts().to_dict())
print("per variety:", wide["Variedad"].value_counts().to_dict())
print("method x variety cells:\n",
      wide.groupby(["Metodo", "Variedad"]).size().unstack().to_string())
print("missing:", wide[FEATS].isna().sum().to_dict())
print("duplicate feature rows:", wide.duplicated(subset=FEATS).sum())
print("process settings per method:")
for m, sub in wide.groupby("Metodo"):
    print("  ", m, "t=", sorted(sub.t.unique()), "T=", sorted(sub["T"].unique()),
          "P=", sorted(sub.P.unique()))

print("\n== 1. HEADLINE REPRODUCTION ==")
b5, f5, n5, _ = run(X, ym, list(StratifiedKFold(5, shuffle=True,
                                                random_state=SEED).split(X, ym)))
rep("random 5-fold (paper 56.7/55.7)", b5, f5, n5)
bl, fl, nl, tel = run(X, ym, list(LeaveOneGroupOut().split(X, ym, g)))
rep("LOVO (paper 38.9/35.9)", bl, fl, nl)
bv, fv, nv, _ = run(X, yv, list(StratifiedKFold(5, shuffle=True,
                                                random_state=SEED).split(X, yv)))
rep("variety task (paper 60.7)", bv, fv, nv)
print("LOVO test sizes:", tel, " 5-fold train sizes:", n5)
print("variety ratio to chance: %.2fx" % (bv.mean() / (1 / 6)))

print("\n== 2. FOLD-COUNT CONFOUND: random CV at k=2..10 ==")
for k in [2, 3, 5, 6, 8, 10]:
    b, f, n, _ = run(X, ym, list(StratifiedKFold(k, shuffle=True,
                                                 random_state=SEED).split(X, ym)))
    print(f"  k={k:>2} train_n={int(np.mean(n)):>3} bal={100*b.mean():5.1f}% "
          f"(+/-{100*b.std():.1f}) F1={100*f.mean():5.1f}%")

print("\n  random 6-fold over 200 seeds (matches LOVO fold count):")
m6 = [run(X, ym, list(StratifiedKFold(6, shuffle=True, random_state=s)
                      .split(X, ym)))[0].mean() for s in range(200)]
m6 = np.array(m6)
print(f"  mean={100*m6.mean():.1f}%  sd={100*m6.std():.1f}  "
      f"2.5-97.5pct=[{100*np.percentile(m6,2.5):.1f},{100*np.percentile(m6,97.5):.1f}]")
m5 = np.array([run(X, ym, list(StratifiedKFold(5, shuffle=True, random_state=s)
                              .split(X, ym)))[0].mean() for s in range(200)])
print(f"  random 5-fold over 200 seeds: mean={100*m5.mean():.1f}% sd={100*m5.std():.1f} "
      f"2.5-97.5pct=[{100*np.percentile(m5,2.5):.1f},{100*np.percentile(m5,97.5):.1f}]")
print(f"  paper's single-seed 56.7% percentile among seeds: "
      f"{100*(m5 < b5.mean()).mean():.0f}th")

print("\n  CONTROL: 6 groups of 27 assigned at random (breaks variety link,")
print("  keeps fold count + grouped structure), 200 draws:")
rng = np.random.default_rng(0)
ctrl = []
for _ in range(200):
    fake = np.repeat(np.arange(6), 27)
    rng.shuffle(fake)
    ctrl.append(run(X, ym, list(LeaveOneGroupOut().split(X, ym, fake)))[0].mean())
ctrl = np.array(ctrl)
print(f"  mean={100*ctrl.mean():.1f}% sd={100*ctrl.std():.1f} "
      f"2.5-97.5pct=[{100*np.percentile(ctrl,2.5):.1f},{100*np.percentile(ctrl,97.5):.1f}]")
print(f"  P(random-group LOVO <= real LOVO {100*bl.mean():.1f}%) = "
      f"{(ctrl <= bl.mean()).mean():.3f}")

print("\n== 3. IS 38.9% ABOVE CHANCE? IS 56.7% vs 38.9% REAL? ==")


def ci_mean(a):
    if len(a) < 2:
        return (np.nan, np.nan)
    se = a.std(ddof=1) / np.sqrt(len(a))
    t = stats.t.ppf(0.975, len(a) - 1)
    return (a.mean() - t * se, a.mean() + t * se)


for tag, a, ch in [("random 5-fold", b5, 1 / 3), ("LOVO", bl, 1 / 3),
                   ("variety", bv, 1 / 6)]:
    lo, hi = ci_mean(a)
    t, p = stats.ttest_1samp(a, ch)
    print(f"  {tag:14} mean={100*a.mean():5.1f}%  95%CI across folds="
          f"[{100*lo:.1f},{100*hi:.1f}]  vs chance {100*ch:.1f}%: "
          f"t={t:.2f} p={p:.4f}  CI_contains_chance={lo <= ch <= hi}")
    print(f"                 mean+/-1sd = [{100*(a.mean()-a.std()):.1f},"
          f"{100*(a.mean()+a.std()):.1f}]  contains chance="
          f"{(a.mean()-a.std()) <= ch <= (a.mean()+a.std())}")

print("\n  pooled-prediction binomial CI (all folds concatenated):")
for tag, splits, y in [("random 5-fold", list(StratifiedKFold(5, shuffle=True,
                                                              random_state=SEED).split(X, ym)), ym),
                       ("LOVO", list(LeaveOneGroupOut().split(X, ym, g)), ym),
                       ("variety 5-fold", list(StratifiedKFold(5, shuffle=True,
                                                               random_state=SEED).split(X, yv)), yv)]:
    pr = np.empty(len(y), dtype=object)
    for tr, te in splits:
        pr[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    pr = np.array(list(pr))
    acc = (pr == y).mean()
    bal_pool = balanced_accuracy_score(y, pr)
    k = int((pr == y).sum()); n = len(y)
    lo, hi = stats.binomtest(k, n).proportion_ci(0.95, method="wilson")
    print(f"  {tag:14} pooled plain acc={100*acc:.1f}% [{100*lo:.1f},{100*hi:.1f}] "
          f"pooled bal_acc={100*bal_pool:.1f}%  ({k}/{n})")

print("\n  permutation test (shuffle method labels, 500 reps):")
for tag, splits_fn in [("random 5-fold", lambda yy: list(StratifiedKFold(
        5, shuffle=True, random_state=SEED).split(X, yy))),
        ("LOVO", lambda yy: list(LeaveOneGroupOut().split(X, yy, g)))]:
    obs = (b5 if tag.startswith("random") else bl).mean()
    null = []
    rr = np.random.default_rng(1)
    for _ in range(500):
        yp = rr.permutation(ym)
        null.append(run(X, yp, splits_fn(yp))[0].mean())
    null = np.array(null)
    pval = (null.sum() * 0 + (null >= obs).sum() + 1) / (len(null) + 1)
    print(f"  {tag:14} obs={100*obs:.1f}%  null mean={100*null.mean():.1f}% "
          f"sd={100*null.std():.1f}  95pct={100*np.percentile(null,95):.1f}%  "
          f"p={pval:.4f}")

print("\n  5-fold vs LOVO: are the two fold distributions distinguishable?")
u = stats.mannwhitneyu(b5, bl, alternative="greater")
tt = stats.ttest_ind(b5, bl, equal_var=False)
print(f"  5-fold folds={np.round(100*b5,1).tolist()}")
print(f"  LOVO   folds={np.round(100*bl,1).tolist()}")
print(f"  Welch t={tt.statistic:.2f} p={tt.pvalue:.4f}   "
      f"Mann-Whitney U p(one-sided)={u.pvalue:.4f}")
print(f"  gap = {100*(b5.mean()-bl.mean()):.1f} points; paper calls it 18")
print("  seed-robust gap: 5-fold over 200 seeds vs fixed LOVO:")
print(f"  P(5fold_seed <= LOVO) = {(m5 <= bl.mean()).mean():.3f}; "
      f"gap mean={100*(m5.mean()-bl.mean()):.1f} pts")

print("\n== 4. VARIETY TASK BALANCE / METRIC ==")
print("variety class counts:", wide["Variedad"].value_counts().to_dict(),
      "-> perfectly balanced, bal_acc == plain acc in expectation")
print(f"60.7% / 16.67% = {bv.mean()/(1/6):.2f}x chance "
      "(paper says 'nearly four times')")
rr = np.random.default_rng(2)
nullv = []
for _ in range(300):
    yp = rr.permutation(yv)
    nullv.append(run(X, yp, list(StratifiedKFold(5, shuffle=True,
                                                 random_state=SEED).split(X, yp)))[0].mean())
nullv = np.array(nullv)
print(f"variety permutation null: mean={100*nullv.mean():.1f}% "
      f"95pct={100*np.percentile(nullv,95):.1f}%  "
      f"p={((nullv>=bv.mean()).sum()+1)/(len(nullv)+1):.4f}")

print("\n== 5. PREPROCESSING DIFFS ==")
print("pivot aggfunc='mean' used; duplicate (keys,C) cells in raw file:",
      int((d.groupby(keys + ['C']).size() > 1).sum()), "-> mean is a no-op")
print("Variedad raw labels (pre-strip):", sorted(d["Variedad"].unique()))
print("strip() merges nothing extra:", d["Variedad"].nunique(),
      "->", wide["Variedad"].nunique())
print("process settings t/T/P are in the pivot INDEX, not in X:",
      "features used =", FEATS)
# does method leak through P?
print("\nLEAK CHECK: can P alone identify method?")
print(wide.groupby("Metodo")["P"].agg(["min", "max", "nunique"]).to_string())
print("\n== 6. WHAT IF process settings were included (not what paper does) ==")
X2 = wide[FEATS + ["t", "T", "P"]].astype(float).to_numpy()
b, f, n, _ = run(X2, ym, list(StratifiedKFold(5, shuffle=True,
                                              random_state=SEED).split(X2, ym)))
rep("5-fold + t/T/P", b, f, n)
