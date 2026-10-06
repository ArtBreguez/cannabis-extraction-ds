"""What the two declared classes actually contain, by product name.

The label is the regulator's category, `non-solvent based` versus
`solvent based`. Under the Nevada rules that govern most of the rows, CO2
extraction counts as non-solvent, so the non-solvent class is wider than the
rosin-and-hash reading of "solventless". This counts, per class, how many
product names carry each family of method or format words. The name regexes
here are descriptive only; they never change a label.

Output: docs/evidence/label_composition.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/evidence/label_composition.txt"

FAMILIES = [
    ("distillate", r"distillate|disty"),
    ("co2 / supercritical", r"\bco2\b|co 2|supercritical"),
    ("ethanol", r"ethanol|etoh"),
    ("hydrocarbon words (bho, butane, live resin, shatter, badder, wax)",
     r"\bbho(?![a-z])|butane|hydrocarbon|live\s*resin|shatter|badder|batter|\bwax\b"),
    ("mechanical words (rosin, hash, kief, bubble, ice water)",
     r"rosin|hash|kief|bubble|ice\s*water|solventless"),
    ("cartridge / vape / pen", r"cartridge|\bcart\b|vape|\bpen\b|disposable"),
]


def main() -> int:
    df = pd.read_csv(ROOT / "data/labeled/master.csv", low_memory=False)
    df = df[df["label_conflict"] == 0].copy()
    name = df["product_name"].fillna("").str.lower()
    out = [f"rows (conflicts excluded): {len(df)}", ""]
    out.append("=== DECLARED product_type VALUES PER CLASS ===")
    for cls in ("solventless", "hydrocarbon"):
        vc = df.loc[df["label"] == cls, "product_type"].value_counts()
        out.append(f"{cls}:")
        out += [f"  {n:>6}  {t}" for t, n in vc.items()]
    out.append("")
    # Only rows that carry a product name can hit a word family: the 4,115
    # producer-less rows have none. Percentages are over the NAMED rows of
    # each class, so a nameless row is not counted as "names nothing".
    named = name.str.strip() != ""
    ns = int((named & (df["label"] == "solventless")).sum())
    nh = int((named & (df["label"] == "hydrocarbon")).sum())
    out.append(f"named rows: {int(named.sum())} of {len(df)}  "
               f"(solventless {ns}, hydrocarbon {nh})")
    out.append("=== PRODUCT-NAME WORD FAMILIES PER CLASS (a name may hit several; % of named rows) ===")
    out.append(f"{'family':66} {'solventless':>12} {'hydrocarbon':>12}")
    for label, pat in FAMILIES:
        hit = name.str.contains(pat, regex=True)
        s = int((hit & (df["label"] == "solventless")).sum())
        h = int((hit & (df["label"] == "hydrocarbon")).sum())
        out.append(f"{label:66} {s:>5} ({100*s/ns:4.1f}%) {h:>5} ({100*h/nh:4.1f}%)")
    out.append("")
    out.append("=== LABORATORIES CARRYING THE NON-SOLVENT + DISTILLATE NAMES ===")
    sd = df[(df["label"] == "solventless") & name.str.contains("distillate|disty")]
    out += [f"  {n:>6}  {lab}" for lab, n in sd["lab"].value_counts().items()]
    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
