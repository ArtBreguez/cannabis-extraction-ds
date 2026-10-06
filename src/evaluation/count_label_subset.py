"""The 139,714 -> 37,374 reduction table in 2.1, counted from the source.

The table was first derived by subtraction, which mixed two populations: the
37,374 labelled rows were counted over every source row, while the 139,714
covers only rows whose product_type mentions a concentrate or extract. This
streams the 2.5 GB file once and counts the labelled rows inside and outside
the concentrate subset separately, using the same NON_SOLVENT pattern as
build_master.py.

Output: docs/evidence/count_label_subset.txt
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.preprocessing.build_master import NON_SOLVENT  # noqa: E402

SRC = ROOT / "data/raw/cannlytics/all-results-latest.csv"
OUT = ROOT / "docs/evidence/count_label_subset.txt"


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    total = conc = declared = undeclared = outside = 0
    undeclared_types: Counter = Counter()
    declared_types: Counter = Counter()
    outside_types: Counter = Counter()
    with open(SRC, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            if len(row) < len(header):
                continue
            total += 1
            pt = (row[idx["product_type"]] or "").strip().lower()
            is_conc = "concentrate" in pt or "extract" in pt
            labelled = bool(NON_SOLVENT.search(pt)) or "solvent based" in pt
            if is_conc:
                conc += 1
                if labelled:
                    declared += 1
                    declared_types[pt] += 1
                else:
                    undeclared += 1
                    undeclared_types[pt] += 1
            elif labelled:
                outside += 1
                outside_types[pt] += 1

    out = [f"source rows:                         {total}",
           f"concentrate or extract rows:         {conc}",
           f"  product_type declares the family:  {declared} "
           f"({100*declared/conc:.1f}%)",
           f"  product_type does not:             {undeclared} "
           f"({100*undeclared/conc:.1f}%)",
           f"labelled rows outside that subset:   {outside}",
           f"total labelled rows:                 {declared + outside}",
           "",
           "declared product_type values:"]
    out += [f"  {n:>7}  {t}" for t, n in declared_types.most_common()]
    out += ["", "labelled values outside the subset:"]
    out += [f"  {n:>7}  {t}" for t, n in outside_types.most_common()]
    out += ["", "largest undeclared product_type values:"]
    out += [f"  {n:>7}  {t}" for t, n in undeclared_types.most_common(8)]
    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
