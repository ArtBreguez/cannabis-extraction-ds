"""Pass twelve, part B: the terpene panel over time, without the confound.

review11_checks.txt reports the share of ALL usable rows with a terpene panel
by year. The producer-less laboratories never report terpenes and their rows
are dated from December 2022 onward, so that series mixes a change in who is
in the data with a change in what is tested. This separates the two.

Output: docs/evidence/review12b_panel.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/evidence/review12b_panel.txt"


def main() -> int:
    df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    has = df["producer"].fillna("").astype(str).str.strip() != ""
    yr = pd.to_datetime(df["date_tested"], errors="coerce").dt.year
    out = ["share of rows with a terpene panel (d_limonene tested), by year of test"]
    out.append(f"{'year':6} {'producer rows':>22} {'producer-less rows':>24}")
    for y in sorted(yr.dropna().unique()):
        a = df[has & (yr == y)]["d_limonene_tested"]
        b = df[~has & (yr == y)]["d_limonene_tested"]
        fa = f"{100*a.mean():5.1f}% of {len(a):>5}" if len(a) else "   -- of     0"
        fb = f"{100*b.mean():5.1f}% of {len(b):>5}" if len(b) else "   -- of     0"
        out.append(f"{int(y):<6} {fa:>22} {fb:>24}")
    und = df[yr.isna()]
    out.append(f"undated rows: {len(und)}  (producer rows {int(has[yr.isna()].sum())}), "
               f"terpene panel on {100*und['d_limonene_tested'].mean():.1f}%")
    for name, col in (("coa_urls", "coa_urls"), ("batch_number", "batch_number"), ("strain_name", "strain_name")):
        f = df[col].notna() & (df[col].astype(str).str.strip() != "")
        out.append(f"usable rows with {name}: {int(f.sum())} of {len(df)}")
    out.append(f"class balance by year, producer rows (solvent-based share):")
    for y in sorted(yr.dropna().unique()):
        a = df[has & (yr == y)]["label"]
        if len(a):
            out.append(f"  {int(y)}: {100*(a == 'hydrocarbon').mean():.1f}% of {len(a)}")
    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
