"""Pass eleven: data problems a third referee found, and the checks they call for.

  T. 1,303 samples in the producer-less laboratories are stored as TWO
     adjacent rows each: same laboratory and date, complementary analytes
     (delta_9_thc in one; cbd, cbda, thca, total_cbd in the other). Every
     all-rows figure inherits the double count; the producer-row figures do
     not, because these rows carry no producer. Measured here, with the
     all-rows figures recomputed after merging each pair into one row.
  Y. the label over time: ethanol-named rows by year and class, and the share
     of rows with a terpene panel by year
  N. what the word families of the 2.1 table actually match
  F. macro-F1 baselines on each scheme's own rows
  L. the logistic comparator with class weighting
  P. a positive control: a target with known chemical signal under the same
     producer hold-out
  R. rows per producer
  B. the producer bootstrap applied to the fold-mean statistic as well as to
     the pooled one

Output: docs/evidence/review11_checks.txt
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.phase4_generalisation import SEED, features  # noqa: E402
from src.evaluation.review10b_robustness import linear_matrix  # noqa: E402

OUT = ROOT / "docs/evidence/review11_checks.txt"


def hgb(lr=0.1, mi=200):
    return HistGradientBoostingClassifier(max_iter=mi, learning_rate=lr,
                                          random_state=SEED, early_stopping=False)


def run(make, X, y, splits, metric=balanced_accuracy_score):
    return np.array([metric(y[te], make().fit(X[tr], y[tr]).predict(X[te]))
                     for tr, te in splits])


def fmt(a):
    return f"{100*a.mean():.1f}% (+/-{100*a.std():.1f})"


def main() -> int:
    full = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = full[full["label_conflict"] == 0].copy()
    feats = features(df)
    out = [f"rows: {len(df)}   features: {len(feats)}"]

    def say(line=""):
        out.append(line)
        print(line, flush=True)

    prod = df["producer"].fillna("").astype(str).str.strip()
    has = (prod != "").to_numpy()
    npres = df[feats].notna().sum(axis=1)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # T ------------------------------------------------------------------
    say()
    say("=== T. samples stored as two rows ===")
    one = set(df.index[(~has) & (npres == 1).to_numpy()])
    four = set(df.index[(~has) & (npres == 4).to_numpy()])
    pairs = [(i, i + 1) for i in sorted(one) if i + 1 in four]
    say(f"producer-less rows: {int((~has).sum())}   with a product name: "
        f"{int((df.loc[~has, 'product_name'].fillna('').astype(str).str.strip() != '').sum())}")
    say(f"rows reporting exactly 1 analyte: {len(one)}   exactly 4: {len(four)}   "
        f"adjacent (1 then 4) pairs: {len(pairs)}")
    a1 = sorted(c for c in feats if df.loc[sorted(one), c].notna().any())
    a4 = sorted(c for c in feats if df.loc[sorted(four), c].notna().any())
    say(f"analytes in the 1-analyte rows: {a1}   in the 4-analyte rows: {a4}")
    for col in ("lab", "date_tested", "label"):
        same = sum(str(full.at[a, col]) == str(full.at[b, col]) for a, b in pairs)
        say(f"pairs with the same {col}: {same} of {len(pairs)}")
    say(f"product_type of the first row: {dict(Counter(full.at[a, 'product_type'] for a, _ in pairs))}")
    say(f"product_type of the second row: {dict(Counter(full.at[b, 'product_type'] for _, b in pairs))}")
    say(f"pairs per laboratory: {dict(Counter(full.at[a, 'lab'] for a, _ in pairs))}")

    merged = df.copy()
    for a, b in pairs:
        for c in feats:
            if pd.isna(merged.at[a, c]) and not pd.isna(merged.at[b, c]):
                merged.at[a, c] = merged.at[b, c]
    merged = merged.drop(index=[b for _, b in pairs])
    say(f"after merging each pair into one row: rows={len(merged)}  "
        f"classes={ {k: int(v) for k, v in merged['label'].value_counts().items()} }  "
        f"majority {100*merged['label'].value_counts(normalize=True).max():.1f}%")

    def patterns(d):
        pat = [tuple(int(v) for v in r) for r in d[feats].notna().to_numpy()]
        by = defaultdict(Counter)
        for p, yy in zip(pat, d["label"]):
            by[p][yy] += 1
        pure = [p for p, c in by.items() if len(c) == 1]
        correct = sum(max(c.values()) for c in by.values())
        return len(by), len(pure), sum(sum(by[p].values()) for p in pure), correct / len(d)

    for name, d in (("as stored", df), ("pairs merged", merged)):
        n, npure, rows_pure, acc = patterns(d)
        say(f"missingness patterns, {name}: {n} distinct, {npure} pure covering {rows_pure} rows "
            f"({100*rows_pure/len(d):.2f}%), in-sample lookup accuracy {100*acc:.1f}% "
            f"vs majority {100*d['label'].value_counts(normalize=True).max():.1f}%")

    Xm = merged[feats].astype(float).to_numpy()
    ym = merged["label"].astype(str).to_numpy()
    r = run(hgb, Xm, ym, list(cv.split(Xm, ym)))
    say(f"random split, all rows, pairs merged: {fmt(r)}")
    labm = merged["lab"].astype(str).to_numpy()
    big = [l for l, c in Counter(labm).items() if c >= 500]
    folds = [(np.where(labm != l)[0], np.where(labm == l)[0]) for l in big
             if len(set(ym[labm == l])) == 2]
    ul = run(hgb, Xm, ym, folds)
    say(f"unseen-lab, pairs merged: folds={len(folds)}  {fmt(ul)}")
    keep = [l for l, c in Counter(labm).items() if c >= 100]
    mk = np.isin(labm, keep)
    base = max(Counter(labm[mk]).values()) / mk.sum()
    lp = run(lambda: hgb(0.05, 120), Xm[mk], labm[mk], list(cv.split(Xm[mk], labm[mk])),
             metric=lambda a, b: float((a == b).mean()))
    say(f"laboratory probe, pairs merged: classes={len(keep)} rows={int(mk.sum())}  "
        f"acc={fmt(lp)}  baseline={100*base:.1f}%  lift={100*(lp.mean()-base):+.1f}pts")

    # Y ------------------------------------------------------------------
    say()
    say("=== Y. the label and the panel over time ===")
    nm = df["product_name"].fillna("").str.lower()
    yr = pd.to_datetime(df["date_tested"], errors="coerce").dt.year
    eth = nm.str.contains("ethanol|etoh")
    tab = pd.crosstab(yr[eth], df["label"][eth])
    say("ethanol-named rows by year (solvent-based / non-solvent):")
    for y_, row in tab.iterrows():
        say(f"  {int(y_)}: {int(row.get('hydrocarbon', 0)):>4} / {int(row.get('solventless', 0)):>4}")
    say("share of rows with a terpene panel (d_limonene tested), by year:")
    t = df.assign(y=yr).groupby("y")["d_limonene_tested"].agg(["mean", "size"])
    for y_, row in t.iterrows():
        say(f"  {int(y_)}: {100*row['mean']:5.1f}%  of {int(row['size'])} rows")

    # N ------------------------------------------------------------------
    say()
    say("=== N. what the name word-families match ===")
    sol = (df["label"] == "solventless")
    hyd = nm.str.contains(r"\bbho(?![a-z])|butane|hydrocarbon|live\s*resin|shatter|badder|batter|\bwax\b")
    mech = nm.str.contains(r"rosin|hash|kief|bubble|ice\s*water|solventless")
    say(f"named rows: {int((nm != '').sum())} of {len(df)}  "
        f"(non-solvent {int(((nm != '') & sol).sum())}, solvent-based {int(((nm != '') & ~sol).sum())})")
    say(f"non-solvent rows hitting a hydrocarbon word: {int((hyd & sol).sum())}, "
        f"of which also say rosin: {int((hyd & sol & nm.str.contains('rosin')).sum())}")
    mh = mech & ~sol
    say(f"solvent-based rows hitting a mechanical word: {int(mh.sum())} "
        f"(rosin {int((mh & nm.str.contains('rosin')).sum())}, hash {int((mh & nm.str.contains('hash')).sum())}, "
        f"bubble {int((mh & nm.str.contains('bubble')).sum())}, kief {int((mh & nm.str.contains('kief')).sum())})")

    # shared matrices for the producer rows
    X = df[feats].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()
    g = prod.to_numpy()
    per = df[has].groupby(prod[has])["label"].nunique()
    dual = has & prod.isin(set(per[per == 2].index)).to_numpy()

    # F ------------------------------------------------------------------
    say()
    say("=== F. macro-F1 baselines on each scheme's own rows ===")
    for name, m in (("unseen-producer rows", has), ("dual-class rows", dual)):
        yy = y[m]
        maj = np.full(len(yy), Counter(yy).most_common(1)[0][0])
        strat = np.random.default_rng(SEED).permutation(yy)
        say(f"  {name}: n={len(yy)}  majority {100*f1_score(yy, maj, average='macro'):.1f}%  "
            f"stratified shuffle {100*f1_score(yy, strat, average='macro'):.1f}%")

    # L ------------------------------------------------------------------
    say()
    say("=== L. logistic comparator, default and class-weighted (producer rows) ===")
    XL = linear_matrix(df, feats)
    for name, make in (("L2, C=1, unweighted", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                       ("L2, C=1, class_weight=balanced", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")))):
        r = run(make, XL[has], y[has], list(cv.split(XL[has], y[has])))
        gg = run(make, XL[has], y[has], list(GroupKFold(5).split(XL[has], y[has], g[has])))
        say(f"  {name:32} random {fmt(r)}   unseen-producer {fmt(gg)}   gap {100*(r.mean()-gg.mean()):.1f} points")

    # P ------------------------------------------------------------------
    say()
    say("=== P. positive control: is a row named distillate? (producer rows) ===")
    yd = np.where(nm.str.contains("distillate|disty").to_numpy(), "distillate", "other")[has]
    r = run(hgb, X[has], yd, list(cv.split(X[has], yd)))
    gg = run(hgb, X[has], yd, list(GroupKFold(5).split(X[has], yd, g[has])))
    say(f"  classes={dict(Counter(yd))}")
    say(f"  random {fmt(r)}   unseen-producer {fmt(gg)}   gap {100*(r.mean()-gg.mean()):.1f} points")

    # R ------------------------------------------------------------------
    say()
    say("=== R. rows per producer ===")
    pc = pd.Series(g[has]).value_counts()
    say(f"  producers={len(pc)}  largest={int(pc.iloc[0])} ({100*pc.iloc[0]/pc.sum():.1f}%)  "
        f"five largest={100*pc.iloc[:5].sum()/pc.sum():.1f}%  median={int(pc.median())}  smallest={int(pc.iloc[-1])}")

    # B ------------------------------------------------------------------
    say()
    say("=== B. producer bootstrap, predictions held fixed, both statistics ===")
    for name, m in (("unseen-producer", has), ("dual-class+unseen", dual)):
        Xs, ys, gs = X[m], y[m], g[m]
        oof = np.empty(len(ys), dtype=object)
        fold = np.empty(len(ys), dtype=int)
        for k, (tr, te) in enumerate(GroupKFold(5).split(Xs, ys, gs)):
            oof[te] = hgb().fit(Xs[tr], ys[tr]).predict(Xs[te])
            fold[te] = k
        oof = oof.astype(str)
        idx_by = defaultdict(list)
        for i, gg_ in enumerate(gs):
            idx_by[gg_].append(i)
        groups = list(idx_by)
        rng = np.random.default_rng(SEED)
        pooled, fmean = [], []
        for _ in range(2000):
            ii = np.concatenate([idx_by[groups[k]] for k in rng.choice(len(groups), len(groups))])
            if len(set(ys[ii])) < 2:
                continue
            pooled.append(balanced_accuracy_score(ys[ii], oof[ii]))
            fs = [balanced_accuracy_score(ys[ii][fold[ii] == k], oof[ii][fold[ii] == k])
                  for k in range(5) if len(set(ys[ii][fold[ii] == k])) == 2]
            fmean.append(np.mean(fs))
        fo = np.mean([balanced_accuracy_score(ys[fold == k], oof[fold == k]) for k in range(5)])
        for stat, obs, v in (("pooled", balanced_accuracy_score(ys, oof), pooled), ("fold mean", fo, fmean)):
            lo, hi = np.percentile(v, [2.5, 97.5])
            say(f"  {name:18} {stat:9} observed {100*obs:.1f}%  95% interval [{100*lo:.1f}, {100*hi:.1f}]  "
                f"draws at or below 50.0%: {100*np.mean(np.array(v) <= 0.5):.1f}%")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
