"""Measure reporting units and cross-validate the two labeling paths.

Two open questions from LEAKAGE_AUDIT.md, both answerable by measurement:

1. UNITS. Labs report concentrations as percent or as mg/g (1% = 10 mg/g).
   Nothing declares which. A silent 10x error is exactly the class of bug that
   passes every check and corrupts the result, so the scale is inferred from
   the distribution of values per lab and compared against physical bounds.

2. LABEL AGREEMENT. DATASET_AUDIT.md defines two label paths: the structured
   product_type (primary, declared at source) and a regex over product_name
   (secondary). Where both exist on the same row they must agree. The
   disagreement rate is the empirical test of the regex rule — if it is high,
   the name-based path does not belong in the training set.

This script only measures. It decides nothing.

Output: docs/evidence/units_and_label_agreement.txt
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/cannlytics/all-results-latest.csv"

NON_SOLVENT = re.compile(r"non[- ]?solvent")

# Ordered: the most specific pattern must win. "live rosin" contains "rosin",
# so rosin must be tested last among the solventless rules.
NAME_RULES = [
    ("hash_rosin", "solventless", re.compile(r"\bhash\s*rosin\b")),
    ("live_rosin", "solventless", re.compile(r"\blive\s*rosin\b")),
    ("rosin", "solventless", re.compile(r"\brosin\b")),
    ("live_resin", "hydrocarbon", re.compile(r"\blive\s*resin\b")),
    ("bho", "hydrocarbon", re.compile(r"\bbho\b|\bbutane\b|\bhydrocarbon\b")),
    ("shatter", "hydrocarbon", re.compile(r"\bshatter\b")),
    ("badder", "hydrocarbon", re.compile(r"\bbadder\b|\bbatter\b")),
    ("wax", "hydrocarbon", re.compile(r"\bwax\b")),
]

# total_thc in a concentrate is physically bounded: roughly 40-99% by weight.
# In percent it lands near 70; in mg/g near 700. The gap is a factor of 10, so
# the median alone separates them unambiguously.
UNIT_PROBE = "total_thc"


def fnum(raw: str):
    v = (raw or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def name_label(name: str):
    for rule, cls, pat in NAME_RULES:
        if pat.search(name):
            return rule, cls
    return None, None


def main() -> int:
    csv.field_size_limit(sys.maxsize)

    thc_by_lab: dict[str, list] = defaultdict(list)
    terp_by_lab: dict[str, list] = defaultdict(list)
    agree = Counter()
    disagree_examples: list[str] = []
    rule_vs_struct: dict[str, Counter] = defaultdict(Counter)
    name_only = Counter()

    with open(SRC, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}

        for row in reader:
            if len(row) < len(header):
                continue
            pt = (row[idx["product_type"]] or "").strip().lower()
            struct = None
            if NON_SOLVENT.search(pt):
                struct = "solventless"
            elif "solvent based" in pt:
                struct = "hydrocarbon"

            is_conc = "concentrate" in pt or "extract" in pt
            if struct is None and not is_conc:
                continue

            lab = (row[idx["lab"]] or "").strip().lower() or "(blank)"
            name = (row[idx["product_name"]] or "").lower()
            rule, by_name = name_label(name)

            if struct:
                v = fnum(row[idx[UNIT_PROBE]])
                if v and v > 0:
                    thc_by_lab[lab].append(v)
                t = fnum(row[idx["total_terpenes"]])
                if t and t > 0:
                    terp_by_lab[lab].append(t)

            if struct and by_name:
                agree["both_labeled"] += 1
                if struct == by_name:
                    agree["agree"] += 1
                    rule_vs_struct[rule]["agree"] += 1
                else:
                    agree["disagree"] += 1
                    rule_vs_struct[rule]["disagree"] += 1
                    if len(disagree_examples) < 12:
                        disagree_examples.append(
                            f"  struct={struct:12} name={by_name:12} "
                            f"rule={rule:11} {name[:52]}"
                        )
            elif struct:
                agree["structured_only"] += 1
            elif by_name:
                name_only[by_name] += 1

    out = []
    out.append("=== REPORTING UNITS PER LAB ===")
    out.append(f"probe: {UNIT_PROBE} in structured-labeled concentrates")
    out.append("a concentrate is ~40-99% THC: median near 70 => percent, "
               "near 700 => mg/g")
    out.append(f"{'lab':32} {'n':>7} {'median':>9} {'p05':>8} {'p95':>8}  unit")
    unit_of = {}
    for lab, vals in sorted(thc_by_lab.items(), key=lambda kv: -len(kv[1])):
        if len(vals) < 20:
            continue
        vals.sort()
        med = statistics.median(vals)
        p05 = vals[int(0.05 * len(vals))]
        p95 = vals[int(0.95 * len(vals))]
        unit = "percent" if med < 100 else "mg/g"
        unit_of[lab] = unit
        out.append(f"{lab[:31]:32} {len(vals):>7} {med:>9.2f} {p05:>8.2f} "
                   f"{p95:>8.2f}  {unit}")
    out.append("")
    out.append(f"distinct units in use: {sorted(set(unit_of.values()))}")

    out.append("")
    out.append("=== TOTAL_TERPENES SANITY ===")
    out.append("terpene totals in concentrates run ~0.5-15%")
    for lab, vals in sorted(terp_by_lab.items(), key=lambda kv: -len(kv[1])):
        if len(vals) < 20:
            continue
        vals.sort()
        out.append(f"{lab[:31]:32} {len(vals):>7} "
                   f"median={statistics.median(vals):>8.2f} "
                   f"p95={vals[int(0.95*len(vals))]:>8.2f}")

    out.append("")
    out.append("=== LABEL AGREEMENT: structured vs product-name regex ===")
    both = agree["both_labeled"]
    out.append(f"rows carrying BOTH labels: {both}")
    if both:
        out.append(f"  agree:    {agree['agree']:>6} "
                   f"({100*agree['agree']/both:.1f}%)")
        out.append(f"  disagree: {agree['disagree']:>6} "
                   f"({100*agree['disagree']/both:.1f}%)")
    out.append(f"structured label only: {agree['structured_only']}")
    out.append(f"name label only (no structured): {dict(name_only)}")

    out.append("")
    out.append("=== PER-RULE RELIABILITY ===")
    out.append("which regex rules are trustworthy, measured not assumed")
    out.append(f"{'rule':14} {'agree':>7} {'disagree':>9} {'accuracy':>9}")
    for rule, c in sorted(rule_vs_struct.items(),
                          key=lambda kv: -sum(kv[1].values())):
        n = c["agree"] + c["disagree"]
        out.append(f"{rule:14} {c['agree']:>7} {c['disagree']:>9} "
                   f"{100*c['agree']/n:>8.1f}%")

    if disagree_examples:
        out.append("")
        out.append("=== SAMPLE DISAGREEMENTS ===")
        out.extend(disagree_examples)

    text = "\n".join(out)
    print(text)
    (ROOT / "docs/evidence/units_and_label_agreement.txt").write_text(
        text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
