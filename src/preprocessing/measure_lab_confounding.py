"""Quantify how much lab identity confounds the extraction-method label.

measure_nondetects.py surfaced a threat: terpene coverage differs sharply by
class (e.g. d_limonene numeric in 40.8% of solventless rows vs 65.6% of
hydrocarbon rows). If that gap comes from WHICH LAB tested the sample rather
than from the extract itself, then a classifier can reach high accuracy by
inferring the lab — and learn nothing about extraction method.

This matters because the roadmap's Gate 4 is an unseen-brand test. A lab
shortcut would sail through a naive split and collapse in the real world.

Two things get measured here:

1. Class mix per lab. If a lab tests almost exclusively one class, then
   "which lab" nearly determines "which class" and the two are entangled.
2. Whether the missingness pattern alone predicts the class. A trivial
   rule built only on WHICH analytes are present — ignoring every measured
   value — is the cheapest possible lab-shaped shortcut. Its accuracy is a
   lower bound on how much leakage is available to a real model.

This script only measures. It decides nothing.

Output: docs/evidence/lab_confounding.txt
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/cannlytics/all-results-latest.csv"

# The models drop the 30 rows whose product_name contradicts product_type
# (build_master.py flags them as label_conflict). An earlier version of this
# probe ran on all 37,374 labelled rows, so its percentages used a different
# denominator from every other figure in the manuscript. Apply the same rule
# here so this log and the model logs describe one population.
sys.path.insert(0, str(ROOT))
from src.preprocessing.build_master import name_label  # noqa: E402

NON_SOLVENT = re.compile(r"non[- ]?solvent")

# The missingness probe MUST see the same columns the model sees, or it is a
# lower bound on the wrong quantity. phase4_generalisation.features() keeps
# every analyte that carries at least one number for a labelled row; that is
# these 19. An earlier version of this script probed only the 6 terpenes
# below the fold, which understated the available shortcut by 4 points.
PROBE_ANALYTES = [
    "alpha_bisabolol", "alpha_humulene", "alpha_pinene", "beta_caryophyllene",
    "beta_myrcene", "beta_pinene", "caryophyllene_oxide", "cbd", "cbda",
    "cbn", "d_limonene", "delta_8_thc", "delta_9_thc", "linalool",
    "terpinolene", "thca", "total_cbd", "total_terpenes", "total_thc",
]


def is_numeric(raw: str) -> bool:
    v = (raw or "").strip()
    if not v:
        return False
    try:
        float(v)
    except ValueError:
        return False
    return True


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(SRC, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            if len(row) < len(header):
                continue
            pt = (row[idx["product_type"]] or "").strip().lower()
            if NON_SOLVENT.search(pt):
                cls = "solventless"
            elif "solvent based" in pt:
                cls = "hydrocarbon"
            else:
                continue
            _, by_name, _ = name_label((row[idx["product_name"]] or "").lower())
            if by_name and by_name != cls:
                continue  # label_conflict, excluded from every model run
            lab = (row[idx["lab"]] or "").strip().lower() or "(blank)"
            producer = (row[idx["producer"]] or "").strip().lower()
            # The missingness fingerprint: which analytes carry a number.
            pattern = tuple(is_numeric(row[idx[t]]) for t in PROBE_ANALYTES
                            if t in idx)
            rows.append((cls, lab, producer, pattern))

    out = []
    out.append(f"labeled rows (conflicts excluded): {len(rows)}")
    total = Counter(c for c, _, _, _ in rows)
    out.append(f"class balance: {dict(total)}")
    majority = max(total.values()) / len(rows)
    out.append(f"majority-class baseline: {100*majority:.1f}%")

    out.append("")
    out.append("=== CLASS MIX PER LAB ===")
    out.append("a lab that tests only one class cannot be separated from it")
    per_lab: dict[str, Counter] = defaultdict(Counter)
    for cls, lab, _, _ in rows:
        per_lab[lab][cls] += 1
    out.append(f"{'lab':32} {'rows':>7} {'solventless':>12} {'hydro':>8} {'skew':>7}")
    entangled = 0
    for lab, c in sorted(per_lab.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        s, h = c["solventless"], c["hydrocarbon"]
        skew = 100 * max(s, h) / n
        if skew >= 90:
            entangled += n
        out.append(f"{lab[:31]:32} {n:>7} {s:>12} {h:>8} {skew:>6.1f}%")
    out.append("")
    out.append(f"rows in labs that are >=90% one class: {entangled} "
               f"({100*entangled/len(rows):.1f}%)")

    out.append("")
    out.append("=== LEAKAGE PROBE: predict class from MISSINGNESS ALONE ===")
    out.append("no measured value is used — only which of the 19 modelled")
    out.append("analytes carry a number, exactly the columns the model sees")
    # Majority vote per missingness pattern. This is the crudest possible
    # shortcut; whatever it scores is available to any model for free.
    by_pattern: dict[tuple, Counter] = defaultdict(Counter)
    for cls, _, _, pattern in rows:
        by_pattern[pattern][cls] += 1
    correct = sum(max(c.values()) for c in by_pattern.values())
    out.append(f"distinct missingness patterns: {len(by_pattern)}")
    out.append(f"accuracy from missingness alone: {100*correct/len(rows):.1f}%")
    out.append(f"(majority-class baseline:         {100*majority:.1f}%)")
    lift = 100 * correct / len(rows) - 100 * majority
    out.append(f"lift over baseline:              {lift:+.1f} points")

    out.append("")
    out.append("=== HOW MUCH OF THAT IS DETERMINISTIC ===")
    out.append("a stratum whose rows are ALL one class hands the label over")
    out.append("outright, before any measured value is read")
    pure_pat = [p for p, c in by_pattern.items() if len(c) == 1]
    pure_rows = sum(sum(by_pattern[p].values()) for p in pure_pat)
    out.append(f"patterns that are 100% one class: {len(pure_pat)} "
               f"of {len(by_pattern)}")
    out.append(f"rows they cover:                  {pure_rows} "
               f"({100*pure_rows/len(rows):.2f}%)")
    for p in sorted(pure_pat, key=lambda q: -sum(by_pattern[q].values()))[:5]:
        c = by_pattern[p]
        cls = next(iter(c))
        n_present = sum(p)
        out.append(f"  n={sum(c.values()):>5}  all {cls:<11} "
                   f"({n_present} of {len(p)} analytes reported)")

    out.append("")
    out.append("=== SAME PROBE, USING LAB IDENTITY ===")
    by_lab_pat: dict[tuple, Counter] = defaultdict(Counter)
    for cls, lab, _, pattern in rows:
        by_lab_pat[(lab,) + pattern][cls] += 1
    correct_lab = sum(max(c.values()) for c in by_lab_pat.values())
    out.append(f"accuracy from lab + missingness:  {100*correct_lab/len(rows):.1f}%")

    out.append("")
    out.append("=== PRODUCERS SPANNING BOTH CLASSES ===")
    prod: dict[str, Counter] = defaultdict(Counter)
    for cls, _, producer, _ in rows:
        if producer:
            prod[producer][cls] += 1
    both = {p: c for p, c in prod.items() if len(c) == 2}
    spanned = sum(sum(c.values()) for c in both.values())
    out.append(f"producers in both classes: {len(both)} of {len(prod)}")
    out.append(f"rows belonging to them:    {spanned} "
               f"({100*spanned/len(rows):.1f}% of labeled data)")
    out.append("this subset is where a model cannot win by memorising producer")

    text = "\n".join(out)
    print(text)
    (ROOT / "docs/evidence/lab_confounding.txt").write_text(text + "\n",
                                                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
