"""Why does my coverage measurement disagree with LEAKAGE_AUDIT.md?

The audit reported d_limonene at 40.8% / 65.6% by class. Measuring "non-empty
string" on master.csv gave 75.7% / 81.1%, and three different analytes gave
IDENTICAL numbers, which cannot be right for independent measurements.

Two candidate explanations:
  1. master.csv filled zeros where the raw file had empties, so "non-empty"
     now counts a real zero AND an imputed one.
  2. The authoritative mask is the *_tested flag, not string emptiness.

This checks both, and reports coverage the correct way.
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path("/home/arthur/cannabis-extraction-ds")
csv.field_size_limit(10_000_000)
rows = list(csv.DictReader((ROOT / "data/labeled/master.csv").open()))

print("  === what is actually in the d_limonene column? ===")
vals = [(r.get("d_limonene") or "").strip() for r in rows]
kind = Counter("EMPTY" if not v
               else "ZERO" if float(v) == 0 else "NUMBER"
               for v in vals)
for k, v in kind.most_common():
    print(f"    {k:<8} {v:>6}")

print("\n  === and the _tested flag? ===")
flag = Counter((r.get("d_limonene_tested") or "").strip() for r in rows)
for k, v in flag.most_common():
    print(f"    {k or '(empty)':<8} {v:>6}")

print("\n  === coverage by class, using the FLAG (authoritative) ===")
print(f"    {'analyte':<24}{'solventless':>12}{'hydrocarbon':>13}{'gap':>8}")
TRUE = {"true", "1", "yes", "True"}
for a in ("d_limonene", "beta_caryophyllene", "alpha_pinene",
          "total_terpenes", "total_thc", "total_cbd"):
    f = a + "_tested"
    if f not in rows[0]:
        continue
    out = []
    for cls in ("solventless", "hydrocarbon"):
        sub = [r for r in rows if (r.get("label") or "").strip() == cls]
        n = sum(1 for r in sub
                if (r.get(f) or "").strip().lower() in TRUE)
        out.append(100 * n / len(sub) if sub else 0.0)
    print(f"    {a:<24}{out[0]:>11.1f}%{out[1]:>12.1f}%{out[1]-out[0]:>+8.1f}")

print("\n  === same thing, counting a NUMBER (zeros excluded) ===")
print("  (this is what the raw-file audit measured)")
print(f"    {'analyte':<24}{'solventless':>12}{'hydrocarbon':>13}{'gap':>8}")
for a in ("d_limonene", "beta_caryophyllene", "alpha_pinene",
          "total_terpenes"):
    if a not in rows[0]:
        continue
    out = []
    for cls in ("solventless", "hydrocarbon"):
        sub = [r for r in rows if (r.get("label") or "").strip() == cls]
        n = 0
        for r in sub:
            v = (r.get(a) or "").strip()
            if v and float(v) != 0:
                n += 1
        out.append(100 * n / len(sub) if sub else 0.0)
    print(f"    {a:<24}{out[0]:>11.1f}%{out[1]:>12.1f}%{out[1]-out[0]:>+8.1f}")

print("\n  === the random-split row, read verbatim from the log ===")
log = (ROOT / "docs/evidence/phase4_generalisation.txt").read_text()
for line in log.splitlines():
    if "balanced_acc" in line:
        print("    " + " ".join(line.split()))
