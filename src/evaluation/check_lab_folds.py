"""How many labs does the unseen-lab scheme actually use, and which are dropped?

The manuscript says "grouped by laboratory, eight folds", which misdescribes
the code. phase4_generalisation.py builds leave-one-lab-out folds over labs
with at least 500 rows, and keeps a fold only if the held-out lab contains both
classes. Eight folds is the outcome of those two filters, not a chosen k.

A referee will ask what was excluded and whether excluding it flatters the
result, so this enumerates it.
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path("/home/arthur/cannabis-extraction-ds")
csv.field_size_limit(10_000_000)
rows = list(csv.DictReader((ROOT / "data/labeled/master.csv").open()))

per = Counter((r.get("lab") or "").strip() for r in rows)
cls = {}
for r in rows:
    lab = (r.get("lab") or "").strip()
    c = (r.get("label") or "").strip()
    if lab and c:
        cls.setdefault(lab, Counter())[c] += 1

big = [(l, n) for l, n in per.most_common() if n >= 500 and l]
small = [(l, n) for l, n in per.most_common() if n < 500 and l]

print(f"  labs total: {len([l for l in per if l])}")
print(f"\n  >= 500 rows ({len(big)} labs, eligible to be held out):")
usable = 0
for l, n in big:
    both = len(cls.get(l, {})) == 2
    if both:
        usable += 1
    h = cls.get(l, {}).get("hydrocarbon", 0)
    print(f"    {l[:32]:<32} n={n:<6} both classes={str(both):<5} "
          f"hydro={100*h/n:.1f}%")

print(f"\n  folds actually produced: {usable} "
      f"(eligible AND containing both classes)")
print(f"  the log reports        : 8")
print(f"  -> {'CONSISTENT' if usable == 8 else 'MISMATCH, investigate'}")

print(f"\n  < 500 rows ({len(small)} labs, excluded):")
tot = 0
for l, n in small:
    tot += n
    print(f"    {l[:32]:<32} n={n}")
print(f"\n  excluded rows: {tot} ({100*tot/len(rows):.2f}% of the data)")
print("  A leave-one-lab-out fold on a 100-row lab would give a test set too")
print("  small to estimate anything, which is the reason for the floor.")
