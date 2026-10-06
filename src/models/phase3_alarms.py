"""Phase 3 alarm models: does the chemistry predict things it should not?

The roadmap puts this BEFORE any method classifier, and the order is not
bureaucratic. If the terpene/cannabinoid profile can identify which lab ran
the test, or which producer made the extract, then a method classifier can
reach a high score by riding that signal and would collapse on unseen brands.

So the shortcuts get measured first, each against its own majority baseline:

  chemistry -> lab       (does the assay fingerprint the laboratory?)
  chemistry -> producer  (does it fingerprint the maker?)

(`state` is also in the probe list but the column is absent from master.csv,
so that probe never runs.)

A model that beats its baseline by a wide margin is an alarm, not a result.

Missing values are passed through as NaN on purpose: HistGradientBoosting
handles them natively, so no imputation decision is smuggled in here. That
also means the analyte PANEL is visible to the probes: a laboratory that never
reports a terpene is identifiable from the NaN pattern alone. The probes
therefore measure instrument and reporting convention together, and a
missingness-only lookup is reported beside them to show how much of the
signal is the panel (57.3% of the laboratory probe's 92.3%). An earlier
version of this header said missingness had been measured at +0.0 points and
was excluded here; both statements were wrong.

Output: docs/evidence/phase3_alarms.txt
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     cross_val_score, cross_validate)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data/labeled/master.csv"
OUT = ROOT / "docs/evidence/phase3_alarms.txt"

SEED = 20260921
PROBE_LR = 0.05


def load():
    df = pd.read_csv(MASTER, low_memory=False)
    # The 30 physically contradictory rows never enter an analysis.
    df = df[df["label_conflict"] == 0].copy()
    return df


def analyte_columns(df: pd.DataFrame) -> list[str]:
    # `date_tested` matches the "_tested" suffix but is a date, not a mask.
    flags = {c for c in df.columns
             if c.endswith("_tested") and c != "date_tested"}
    cols = [c[: -len("_tested")] for c in flags]
    # Keep only columns that actually carry numbers; a column that is never
    # populated is not a feature, it is noise with a name.
    keep = []
    for c in cols:
        if c in df.columns and df[c].notna().sum() > 0:
            keep.append(c)
    return sorted(keep)


def probe(df: pd.DataFrame, feats: list[str], target: str,
          min_count: int, out: list[str]) -> None:
    """Can the chemistry predict `target`? Compare against always-guessing."""
    sub = df[df[target].notna() & (df[target].astype(str).str.strip() != "")]
    counts = sub[target].value_counts()
    keep = counts[counts >= min_count].index
    sub = sub[sub[target].isin(keep)]

    if sub[target].nunique() < 2 or len(sub) < 200:
        out.append(f"{target}: not enough data ({len(sub)} rows, "
                   f"{sub[target].nunique()} classes) — skipped")
        return

    X = sub[feats].astype(float).to_numpy()
    y = sub[target].astype(str).to_numpy()

    baseline = counts[keep].max() / len(sub)
    # learning_rate=0.05, not the 0.1 default: with 11 or 33 perfectly
    # separable classes the softmax boosting diverges at 0.1 (fold accuracies
    # 40.6 to 69.4 for the lab, 10.4 to 68.7 for the producer), which an
    # earlier version of this log reported as a +15.3-point lift. The binary
    # task in phase 4 is unaffected (fold spread 0.4) and keeps the default.
    clf = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=PROBE_LR, random_state=SEED,
        early_stopping=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = cross_validate(clf, X, y, cv=cv, n_jobs=-1,
                         scoring=("accuracy", "balanced_accuracy"))
    scores = res["test_accuracy"]
    bal = res["test_balanced_accuracy"]

    acc = scores.mean()
    lift = 100 * (acc - baseline)
    verdict = ("ALARM" if lift > 15 else
               "watch" if lift > 5 else "ok")
    out.append(f"{target:10} classes={sub[target].nunique():>4} "
               f"rows={len(sub):>6}  acc={100*acc:>5.1f}% "
               f"(+/-{100*scores.std():.1f})  baseline={100*baseline:>5.1f}%  "
               f"lift={lift:>+6.1f}pts  [{verdict}]")
    out.append(f"{'':10} folds={[round(100*float(v), 1) for v in scores]}  "
               f"balanced_acc={100*bal.mean():.1f}% (+/-{100*bal.std():.1f}) "
               f"chance={100/sub[target].nunique():.1f}%")


def main() -> int:
    df = load()
    feats = analyte_columns(df)

    out: list[str] = []
    out.append(f"rows (conflicts excluded): {len(df)}")
    out.append(f"analyte features used:     {len(feats)}")
    out.append(f"features: {', '.join(feats)}")
    out.append("")
    out.append("=== ALARM MODELS: chemistry -> metadata ===")
    out.append("a wide lift means the profile fingerprints that field, and any")
    out.append("later method result would be suspect until proven otherwise")
    out.append("")

    for target, min_count in [("lab", 100), ("state", 100),
                              ("producer", 150)]:
        if target in df.columns:
            probe(df, feats, target, min_count, out)

    # --- why 0.05: document the divergence at the default rate ------------
    out.append("")
    out.append("=== WHY learning_rate=0.05: the lab probe at the 0.1 default ===")
    sub = df[df["lab"].notna() & (df["lab"].astype(str).str.strip() != "")]
    counts = sub["lab"].value_counts()
    sub = sub[sub["lab"].isin(counts[counts >= 100].index)]
    X = sub[feats].astype(float).to_numpy()
    y = sub["lab"].astype(str).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for lr in (0.1, PROBE_LR):
        sc = cross_val_score(HistGradientBoostingClassifier(
            max_iter=120, learning_rate=lr, random_state=SEED,
            early_stopping=False), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        out.append(f"  learning_rate={lr:<5} folds={[round(100*float(v), 1) for v in sc]}"
                   f"  mean={100*sc.mean():.1f}%")
    sp_ = df[df["producer"].notna() & (df["producer"].astype(str).str.strip() != "")]
    cp_ = sp_["producer"].value_counts()
    sp_ = sp_[sp_["producer"].isin(cp_[cp_ >= 150].index)]
    sc = cross_val_score(HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.1, random_state=SEED,
        early_stopping=False), sp_[feats].astype(float).to_numpy(),
        sp_["producer"].astype(str).to_numpy(), cv=cv, scoring="accuracy",
        n_jobs=-1)
    out.append(f"  producer probe at 0.1: folds={[round(100*float(v), 1) for v in sc]}"
               f"  mean={100*sc.mean():.1f}%  lift={100*(sc.mean()-cp_[cp_>=150].max()/len(sp_)):+.1f}pts")

    # --- missingness-only lab probe --------------------------------------
    out.append("")
    out.append("=== LAB FROM THE MISSINGNESS PATTERN ALONE (no measured value) ===")
    pat = pd.Series([tuple(int(v) for v in r)
                     for r in sub[feats].notna().to_numpy()], index=sub.index)
    hits = 0
    for tr, te in cv.split(X, y):
        vote = (pd.DataFrame({"p": pat.iloc[tr].to_numpy(), "y": y[tr]})
                .groupby("p")["y"].agg(lambda c: c.value_counts().index[0]))
        pred = pat.iloc[te].map(vote).fillna(pd.Series(y[tr]).mode()[0])
        hits += int((pred.to_numpy() == y[te]).sum())
    out.append(f"  acc={100*hits/len(y):.1f}%  baseline={100*counts[counts>=100].max()/len(y):.1f}%"
               f"  (lookup table over {pat.nunique()} patterns, same 5 folds)")

    # --- lab probe with producers held out --------------------------------
    out.append("")
    out.append("=== LAB PROBE, PRODUCER-GROUPED (no producer in both train and test) ===")
    prod = df["producer"].fillna("").astype(str).str.strip()
    sp = df[(prod != "") & df["lab"].notna()]
    counts = sp["lab"].value_counts()
    sp = sp[sp["lab"].isin(counts[counts >= 100].index)]
    Xp = sp[feats].astype(float).to_numpy()
    yp = sp["lab"].astype(str).to_numpy()
    gp = prod[sp.index].to_numpy()
    sc = cross_val_score(HistGradientBoostingClassifier(
        max_iter=120, learning_rate=PROBE_LR, random_state=SEED,
        early_stopping=False), Xp, yp, groups=gp, cv=GroupKFold(5),
        scoring="accuracy", n_jobs=-1)
    out.append(f"  classes={sp['lab'].nunique()} rows={len(sp)}  "
               f"acc={100*sc.mean():.1f}% (+/-{100*sc.std():.1f})  "
               f"baseline={100*counts[counts>=100].max()/len(sp):.1f}%  "
               f"folds={[round(100*float(v), 1) for v in sc]}")

    out.append("")
    out.append("=== REFERENCE: the actual task, naive split ===")
    out.append("not a result — just the number an alarm would have to explain")
    X = df[feats].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()
    baseline = pd.Series(y).value_counts().max() / len(y)
    clf = HistGradientBoostingClassifier(
        max_iter=120, random_state=SEED, early_stopping=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sc = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy",
                         n_jobs=-1)
    out.append(f"label      rows={len(y):>6}  balanced_acc={100*sc.mean():>5.1f}%"
               f" (+/-{100*sc.std():.1f})  majority={100*baseline:>5.1f}%")
    out.append("")
    out.append("A random-split score is optimistic by construction: the same")
    out.append("producer appears in train and test. Phase 4 replaces it with")
    out.append("GroupKFold and an unseen-brand split.")

    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
