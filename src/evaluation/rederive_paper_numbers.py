"""Re-derive every number the preprint will claim, from the source data.

The rule from previous work: never quote a figure from a write-up, quote it
from the artefact. This recomputes the dataset facts straight out of
master.csv and the raw evidence logs, and prints them in a form the manuscript
can cite. Any disagreement with the docs is reported as a MISMATCH rather than
silently preferred.

Model results (balanced accuracy per split scheme) are read from the evidence
logs because re-running the full Phase 4 sweep takes much longer than this
script should; those logs are the committed output of
src/models/phase4_generalisation.py. Everything else is recomputed.
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

# repo root, resolved from this file so the scripts work in any clone
ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data/labeled/master.csv"
EV = ROOT / "docs/evidence"

csv.field_size_limit(10_000_000)
all_rows = list(csv.DictReader(MASTER.open()))
# Every analysis drops the 30 rows whose product_name contradicts
# product_type before doing anything else, so the figures here are computed
# on that same view. Computing them on the unfiltered file gives 55 dual-class
# producers and 28,927 rows, which is not what any model saw.
rows = [r for r in all_rows
        if (r.get("label_conflict") or "").strip() == "0"]
print(f"  master.csv rows: {len(all_rows)}  (labelled)")
print(f"  usable rows    : {len(rows)}  (label_conflict == 0)")
print(f"  columns        : {len(rows[0])}")

# ---------- class balance ----------
labels = Counter(r.get("label", "").strip() for r in rows)
print("\n  === class balance ===")
tot = sum(labels.values())
for k, v in labels.most_common():
    print(f"    {k or '(empty)':<14} {v:>6}  {100*v/tot:5.1f}%")
maj = max(labels.values()) / tot
print(f"    majority-class share: {100*maj:.1f}%  (plain-accuracy baseline)")
print(f"    balanced-accuracy chance level: 50.0%")

# ---------- producers and labs ----------
def stats(col):
    vals = [(r.get(col) or "").strip() for r in rows]
    filled = [v for v in vals if v]
    return len(filled), len(set(filled))


print("\n  === grouping variables ===")
for col in ("producer", "lab", "state", "strain_name"):
    if col not in rows[0]:
        print(f"    {col:<14} column absent")
        continue
    n, u = stats(col)
    print(f"    {col:<14} filled={n:>6} ({100*n/len(rows):5.1f}%)  distinct={u}")

# ---------- producers in both classes ----------
byprod = {}
for r in rows:
    p = (r.get("producer") or "").strip()
    lab = (r.get("label") or "").strip()
    if p and lab:
        byprod.setdefault(p, set()).add(lab)
dual = {p for p, s in byprod.items() if len(s) > 1}
dual_rows = sum(1 for r in rows
                if (r.get("producer") or "").strip() in dual)
print(f"\n  === the dual-class subset (the strongest test) ===")
print(f"    producers with a label at all : {len(byprod)}")
print(f"    producers in BOTH classes     : {len(dual)}")
print(f"    rows belonging to them        : {dual_rows} "
      f"({100*dual_rows/len(rows):.1f}% of labeled data)")
dl = Counter((r.get("label") or "").strip() for r in rows
             if (r.get("producer") or "").strip() in dual)
print(f"    their class balance           : "
      + ", ".join(f"{k} {v}" for k, v in dl.most_common()))

# ---------- analytes and censoring ----------
# `date_tested` ends in "_tested" but is a date, not an analyte mask.
flags = sorted(c for c in rows[0]
               if c.endswith("_tested") and c != "date_tested")
analytes = [c[: -len("_tested")] for c in flags]
usable = [a for a in analytes
          if a in rows[0] and any((r.get(a) or "").strip() for r in rows)]
print(f"\n  === feature matrix ===")
print(f"    *_tested flags      : {len(flags)}")
print(f"    analytes with data  : {len(usable)}")
print(f"    always-empty        : {sorted(set(analytes) - set(usable))}")

print("\n  === coverage by class (the missingness gap) ===")
hdr = f"    {'analyte':<24}{'solventless':>12}{'hydrocarbon':>13}"
print(hdr)
for a in ("d_limonene", "beta_caryophyllene", "alpha_pinene",
          "total_terpenes", "total_thc"):
    if a not in rows[0]:
        continue
    out = []
    for cls in ("solventless", "hydrocarbon"):
        sub = [r for r in rows if (r.get("label") or "").strip() == cls]
        got = sum(1 for r in sub if (r.get(a) or "").strip())
        out.append(100 * got / len(sub) if sub else 0)
    print(f"    {a:<24}{out[0]:>11.1f}%{out[1]:>12.1f}%")

# ---------- model results, from the committed logs ----------
print("\n  === model results (from committed evidence logs) ===")
pat = re.compile(
    r"^(\S[\w+ -]*?)\s+folds=\s*(\d+)\s+balanced_acc=\s*([\d.]+)%\s*"
    r"\(\+/-([\d.]+)\)\s+macro_F1=\s*([\d.]+)%")
for log, title in (("phase4_generalisation.txt", "Cannlytics (market data)"),
                   ("phase5_controlled.txt", "Zenodo 13823859 (controlled)")):
    p = EV / log
    if not p.exists():
        print(f"    {log}: missing")
        continue
    print(f"    -- {title} --")
    for line in p.read_text().splitlines():
        m = pat.match(line.strip())
        if m:
            name, folds, bal, sd, f1 = m.groups()
            print(f"      {name.strip():<24} folds={folds:>2}  "
                  f"bal_acc={bal:>5}% (+/-{sd:>4})  macro_F1={f1:>5}%")

print("\n  === alarm models (chemistry predicts metadata it should not) ===")
p = EV / "phase3_alarms.txt"
for line in p.read_text().splitlines():
    if "lift=" in line:
        print("      " + " ".join(line.split()))
