"""Phase 3 alarm models: does the chemistry predict things it should not?

The roadmap puts this BEFORE any method classifier, and the order is not
bureaucratic. If the terpene/cannabinoid profile can identify which lab ran
the test, or which producer made the extract, then a method classifier can
reach a high score by riding that signal and would collapse on unseen brands.

So the shortcuts get measured first, each against its own majority baseline:

  chemistry -> lab       (does the assay fingerprint the instrument?)
  chemistry -> state     (does it fingerprint the market?)
  chemistry -> producer  (does it fingerprint the maker?)

A model that beats its baseline by a wide margin is an alarm, not a result.

Missing values are passed through as NaN on purpose: HistGradientBoosting
handles them natively, so no imputation decision is smuggled in here. The
`*_tested` flags are deliberately EXCLUDED from these probes — missingness was
already measured (LEAKAGE_AUDIT.md, +0.0 points) and including it again would
conflate "the chemistry identifies the lab" with "the missing-value pattern
identifies the lab", which are different claims.

Output: docs/evidence/phase3_alarms.txt
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data/labeled/master.csv"
OUT = ROOT / "docs/evidence/phase3_alarms.txt"

SEED = 20260921


def load():
    df = pd.read_csv(MASTER, low_memory=False)
    # The 30 physically contradictory rows never enter an analysis.
    df = df[df["label_conflict"] == 0].copy()
    return df


def analyte_columns(df: pd.DataFrame) -> list[str]:
    flags = {c for c in df.columns if c.endswith("_tested")}
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
    clf = HistGradientBoostingClassifier(
        max_iter=120, random_state=SEED, early_stopping=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    acc = scores.mean()
    lift = 100 * (acc - baseline)
    verdict = ("ALARM" if lift > 15 else
               "watch" if lift > 5 else "ok")
    out.append(f"{target:10} classes={sub[target].nunique():>4} "
               f"rows={len(sub):>6}  acc={100*acc:>5.1f}% "
               f"(+/-{100*scores.std():.1f})  baseline={100*baseline:>5.1f}%  "
               f"lift={lift:>+6.1f}pts  [{verdict}]")


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
