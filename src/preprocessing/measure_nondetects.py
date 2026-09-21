"""Measure how non-detects are encoded, before deciding how to treat them.

The roadmap calls this the first decision of Phase 2, and it is not cosmetic:
`ND`, `<LOQ`, an empty cell and a literal `0` mean different things. Collapsing
all of them to zero makes "this lab does not test terpene X" indistinguishable
from "this product contains no terpene X". Since lab correlates with market and
with extraction method, that would inject leakage that looks like signal.

This script only measures. It decides nothing.

Output: docs/evidence/nondetect_encoding.txt
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/cannlytics/all-results-latest.csv"

# The two label paths from DATASET_AUDIT.md. Structured is the primary one.
NON_SOLVENT = re.compile(r"non[- ]?solvent")

# A spread of analytes: majors, common terpenes, and a rarer one. The rare
# analyte matters most — that is where "not tested" hides.
ANALYTES = [
    "total_thc", "total_cbd", "total_terpenes",
    "thca", "cbd", "cbda", "delta_9_thc",
    "beta_myrcene", "d_limonene", "beta_caryophyllene",
    "alpha_pinene", "beta_pinene", "alpha_terpinene",
    "caryophyllene_oxide",
]


def classify(raw: str) -> str:
    """Bucket a raw cell into the distinctions that matter."""
    v = raw.strip()
    if v == "":
        return "empty"
    low = v.lower()
    if low in ("nd", "n/d", "none detected", "not detected"):
        return "ND"
    if low.startswith("<") or "loq" in low or "lod" in low:
        return "<LOQ/<LOD"
    if low in ("na", "n/a", "nan", "null", "none"):
        return "NA"
    try:
        f = float(v)
    except ValueError:
        return f"other:{v[:12]}"
    return "zero" if f == 0 else "numeric"


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    with open(SRC, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}

        missing = [a for a in ANALYTES if a not in idx]
        cols = [a for a in ANALYTES if a in idx]

        # per analyte -> encoding counts, split by class
        enc: dict[str, Counter] = defaultdict(Counter)
        # per lab -> how many analytes it ever reports a number for
        lab_numeric: dict[str, set] = defaultdict(set)
        lab_rows: Counter = Counter()
        n = 0

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
            n += 1
            lab = (row[idx["lab"]] or "").strip().lower() or "(blank)"
            lab_rows[lab] += 1
            for a in cols:
                kind = classify(row[idx[a]])
                enc[a][f"{cls}/{kind}"] += 1
                enc[a][kind] += 1
                if kind == "numeric":
                    lab_numeric[lab].add(a)

    out = []
    out.append(f"labeled rows analysed: {n}")
    if missing:
        out.append(f"columns absent from the file: {missing}")
    out.append("")
    out.append("=== HOW EACH ANALYTE IS ENCODED ===")
    out.append(f"{'analyte':22} {'numeric':>8} {'empty':>8} {'zero':>7} "
               f"{'ND':>6} {'<LOQ':>6} {'other':>7}")
    for a in cols:
        c = enc[a]
        other = sum(v for k, v in c.items()
                    if "/" not in k and k.startswith(("other:", "NA")))
        out.append(f"{a:22} {c['numeric']:>8} {c['empty']:>8} {c['zero']:>7} "
                   f"{c['ND']:>6} {c['<LOQ/<LOD']:>6} {other:>7}")

    out.append("")
    out.append("=== COVERAGE BY CLASS (numeric share) ===")
    out.append(f"{'analyte':22} {'solventless':>12} {'hydrocarbon':>12}")
    for a in cols:
        c = enc[a]
        s_tot = sum(v for k, v in c.items() if k.startswith("solventless/"))
        h_tot = sum(v for k, v in c.items() if k.startswith("hydrocarbon/"))
        s = 100 * c["solventless/numeric"] / s_tot if s_tot else 0
        h = 100 * c["hydrocarbon/numeric"] / h_tot if h_tot else 0
        out.append(f"{a:22} {s:>11.1f}% {h:>11.1f}%")

    out.append("")
    out.append("=== PER-LAB ANALYTE COVERAGE ===")
    out.append("a lab that never reports an analyte is 'not tested', not 'absent'")
    out.append(f"{'lab':34} {'rows':>7} {'analytes reported':>18}")
    for lab, rows in lab_rows.most_common():
        out.append(f"{lab[:33]:34} {rows:>7} "
                   f"{len(lab_numeric[lab]):>13}/{len(cols)}")

    text = "\n".join(out)
    print(text)
    dest = ROOT / "docs/evidence/nondetect_encoding.txt"
    dest.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
