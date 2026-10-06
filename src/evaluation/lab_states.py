"""Which state each laboratory reports from, and the date range of the rows.

master.csv does not carry `lab_state`, so the manuscript could not say which
markets the two declared categories come from. This streams the raw file once
with the same labelling and conflict rules as build_master.py and tabulates
lab x lab_state over the usable rows, plus the span of `date_tested`.

Output: docs/evidence/lab_states.txt
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.preprocessing.build_master import NON_SOLVENT, name_label  # noqa: E402

SRC = ROOT / "data/raw/cannlytics/all-results-latest.csv"
OUT = ROOT / "docs/evidence/lab_states.txt"


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    by_lab: dict[str, Counter] = defaultdict(Counter)
    by_state: Counter = Counter()
    dates = []
    n = 0
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
                continue
            n += 1
            lab = (row[idx["lab"]] or "").strip() or "(blank)"
            st = (row[idx["lab_state"]] or "").strip() or "(blank)"
            by_lab[lab][st] += 1
            by_state[st] += 1
            d = (row[idx["date_tested"]] or "").strip()[:10]
            if d:
                dates.append(d)
    out = [f"usable rows: {n}", "", "=== lab_state per laboratory ==="]
    for lab, c in sorted(by_lab.items(), key=lambda kv: -sum(kv[1].values())):
        out.append(f"  {sum(c.values()):>6}  {lab:32} {dict(c)}")
    out += ["", "=== rows per lab_state ==="]
    out += [f"  {v:>6}  {k}" for k, v in by_state.most_common()]
    out += ["", f"date_tested filled: {len(dates)} of {n}",
            f"date_tested range: {min(dates)} to {max(dates)}"]
    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
