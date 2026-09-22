"""Build the master dataset: one row per sample, provenance on every row.

Inputs are never modified. This reads data/raw/ and writes data/labeled/.

Design decisions, each traceable to a measurement in docs/:

- Label comes from product_type (declared at source). Where the product name
  also carries a method, the two agree 99.3% of the time
  (units_and_label_agreement.txt), so the name path is used only to FLAG
  conflicts, never to overwrite.

- Conflicting rows are kept but marked `label_conflict`, not silently
  dropped. 68 rows say "non-solvent based" while the product name says live
  resin/BHO/shatter — physically contradictory, since live resin is by
  definition hydrocarbon-extracted. Somebody mislabeled at source, and we
  cannot tell which side is wrong, so they are excluded from the training
  view by default and left visible for audit.

- Every analyte gets a value column and a `*_tested` flag. An empty cell is
  not a zero: 'not tested' and 'measured at zero' are different facts
  (LEAKAGE_AUDIT.md).

- alpha_terpinene is dropped: empty in 100% of labeled rows.

- No unit conversion is applied. All 12 labs report percent
  (units_and_label_agreement.txt); a conversion here would be a silent 10x
  bug waiting to happen.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/cannlytics/all-results-latest.csv"
DEST = ROOT / "data/labeled/master.csv"

NON_SOLVENT = re.compile(r"non[- ]?solvent")

NAME_RULES = [
    # Order matters and is not arbitrary. Texture words (badder, wax, shatter)
    # describe CONSISTENCY, not extraction method, so an explicit method word
    # anywhere in the name outranks them.
    # Explicit solvent naming beats texture: "gelato 33 badder bho100223"
    # names both, and the solvent is the stronger evidence.
    # \b...\b would miss "bho100223" (a batch code glued to the token), which
    # appears in real product names. The negative lookahead accepts a digit or
    # separator after "bho" but still rejects words like "bhopal".
    ("bho", "hydrocarbon", "high",
     re.compile(r"\bbho(?![a-z])|\bbutane\b|\bhydrocarbon\b")),
    # Solventless method words. Must precede the texture rules below, because
    # "live rosin badder" is rosin that happens to have badder consistency —
    # 41 rows in this file are exactly that.
    ("hash_rosin", "solventless", "high", re.compile(r"\bhash\s*rosin\b")),
    ("live_rosin", "solventless", "high", re.compile(r"\blive\s*rosin\b")),
    ("rosin", "solventless", "high", re.compile(r"\brosin\b")),
    # "Live resin" is by definition hydrocarbon-extracted from fresh-frozen
    # material; the term is not used for solventless products.
    ("live_resin", "hydrocarbon", "high", re.compile(r"\blive\s*resin\b")),
    # Texture only. Hydrocarbon is the common case, but a solventless product
    # can share the texture, so these are medium confidence at best.
    ("shatter", "hydrocarbon", "medium", re.compile(r"\bshatter\b")),
    ("badder", "hydrocarbon", "medium", re.compile(r"\bbadder\b|\bbatter\b")),
    ("wax", "hydrocarbon", "medium", re.compile(r"\bwax\b")),
]

# alpha_terpinene deliberately absent: never populated in labeled rows.
CANNABINOIDS = ["total_thc", "total_cbd", "thca", "cbd", "cbda", "delta_9_thc",
                "delta_8_thc", "cbn", "cbg", "cbga", "cbc", "thcv"]
TERPENES = ["total_terpenes", "beta_myrcene", "d_limonene",
            "beta_caryophyllene", "alpha_pinene", "beta_pinene",
            "caryophyllene_oxide", "linalool", "alpha_humulene",
            "terpinolene", "ocimene", "alpha_bisabolol", "camphene",
            "eucalyptol", "geraniol", "guaiol", "isopulegol", "nerolidol",
            "beta_ocimene", "valencene", "fenchol", "borneol", "terpineol"]

META = ["product_name", "product_type", "producer", "brand", "lab", "state",
        "strain_name", "batch_number", "date_tested", "coa_urls"]


def fnum(raw: str):
    v = (raw or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def name_label(name: str):
    for rule, cls, confidence, pat in NAME_RULES:
        if pat.search(name):
            return rule, cls, confidence
    return None, None, None


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    DEST.parent.mkdir(parents=True, exist_ok=True)

    with open(SRC, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}

        analytes = [a for a in CANNABINOIDS + TERPENES if a in idx]
        meta = [m for m in META if m in idx]

        out_cols = (["label", "label_source", "label_conflict",
                     "name_rule", "name_label", "name_confidence"]
                    + meta
                    + analytes
                    + [f"{a}_tested" for a in analytes])

        stats = {"rows": 0, "kept": 0, "conflict": 0,
                 "solventless": 0, "hydrocarbon": 0}

        with open(DEST, "w", newline="", encoding="utf-8") as out:
            w = csv.writer(out)
            w.writerow(out_cols)

            for row in reader:
                stats["rows"] += 1
                if len(row) < len(header):
                    continue
                pt = (row[idx["product_type"]] or "").strip().lower()
                if NON_SOLVENT.search(pt):
                    label = "solventless"
                elif "solvent based" in pt:
                    label = "hydrocarbon"
                else:
                    continue

                name = (row[idx["product_name"]] or "").lower()
                rule, by_name, confidence = name_label(name)
                conflict = bool(by_name and by_name != label)

                rec = [label, "product_type", int(conflict),
                       rule or "", by_name or "", confidence or ""]
                rec += [(row[idx[m]] or "").strip() for m in meta]

                vals, flags = [], []
                for a in analytes:
                    v = fnum(row[idx[a]])
                    # Empty stays empty; a measured zero stays zero.
                    vals.append("" if v is None else v)
                    flags.append(int(v is not None))
                rec += vals + flags

                w.writerow(rec)
                stats["kept"] += 1
                stats[label] += 1
                if conflict:
                    stats["conflict"] += 1

    print(f"scanned rows:      {stats['rows']}")
    print(f"written:           {stats['kept']}")
    print(f"  solventless:     {stats['solventless']}")
    print(f"  hydrocarbon:     {stats['hydrocarbon']}")
    print(f"  label_conflict:  {stats['conflict']} "
          f"(flagged, excluded from the default training view)")
    print(f"analyte columns:   {len(analytes)} (+{len(analytes)} _tested flags)")
    print(f"dest:              {DEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
