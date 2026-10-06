"""Second adversarial pass: internal consistency, not arithmetic.

The first pass encoded referee objections and the audit checks figures against
source. Neither can see a sentence that contradicts another sentence, or a
count that the Results section does not deliver on. Reading the text turned up
four candidates; this verifies each against the data and the logs.
"""
import csv
import re
from collections import Counter
from pathlib import Path

# repo root, resolved from this file so the scripts work in any clone
ROOT = Path(__file__).resolve().parents[2]
T = (ROOT / "PREPRINT.md").read_text()
EV = ROOT / "docs/evidence"
csv.field_size_limit(10_000_000)
rows = list(csv.DictReader((ROOT / "data/labeled/master.csv").open()))

print("  === 1. Methods promises 'three probes'. How many does 3.1 report? ===")
promised = re.search(r"(three|two|four) probes asked whether", T)
print(f"    text says: {promised.group(1) if promised else '?'} probes")
log3 = (EV / "phase3_alarms.txt").read_text()
alarm_lines = [l for l in log3.splitlines() if "lift=" in l]
print(f"    alarm lines in the log : {len(alarm_lines)}")
for l in alarm_lines:
    print(f"      {' '.join(l.split())}")
res31 = T.split("### 3.1")[1].split("### 3.2")[0]
reported = len(re.findall(r"^(lab|producer|label|strain)\s", res31, re.M))
print(f"    targets shown in 3.1   : {reported}")
print(f"    -> {'CONSISTENT' if reported == len(alarm_lines) else 'MISMATCH'}")

print("\n  === 2. Does 2.3 still call the SMALL gap 'a gap of that size'? ===")
sec = T.split("### 2.3")[1].split("### 2.4")[0]
first = sec.strip().split("\n")[0]
print(f"    opening line: {first}")
print("    2.2 now says the TESTED gap is 1-6 points and the DETECTED gap")
print("    is 18-25. 'A coverage gap of that size is a candidate shortcut'")
print("    points at the small one, which is not the motivating number.")
print(f"    -> {'needs rewording' if 'coverage gap of that size' in sec else 'ok'}")

print("\n  === 3. Verify the 373 rows / 1.0% claim ===")
bylab = {}
for r in rows:
    lab = (r.get("lab") or "").strip()
    cls = (r.get("label") or "").strip()
    if lab and cls:
        bylab.setdefault(lab, Counter())[cls] += 1
skewed = {k: v for k, v in bylab.items()
          if max(v.values()) / sum(v.values()) >= 0.90}
nrows = sum(sum(v.values()) for v in skewed.values())
print(f"    labs >=90% one class : {len(skewed)}")
print(f"    rows in them         : {nrows} ({100*nrows/len(rows):.1f}%)")
print(f"    text claims          : 373 rows, 1.0%")
print(f"    -> {'CONFIRMED' if nrows == 373 else 'MISMATCH'}")
for k, v in sorted(skewed.items(), key=lambda x: -sum(x[1].values())):
    tot = sum(v.values())
    print(f"      {k[:34]:<34} n={tot:<5} {max(v.values())/tot:.0%} one class")

print("\n  === 4. Two largest labs: are the counts right? ===")
top = sorted(bylab.items(), key=lambda x: -sum(x[1].values()))[:2]
for k, v in top:
    tot = sum(v.values())
    h = v.get("hydrocarbon", 0)
    print(f"    {k[:30]:<30} n={tot:<6} hydro {100*h/tot:.1f}% "
          f"(overall 65.8%)")
print(f"    text claims 15,035 and 12,539")
print(f"    measured    {sum(top[0][1].values())} and {sum(top[1][1].values())}")

print("\n  === 5. unseen-lab uses 8 folds but there are 12 labs. Why? ===")
print(f"    distinct labs in data: {len(bylab)}")
m = re.search(r"unseen-lab\s+folds=\s*(\d+)", (EV / 'phase4_generalisation.txt').read_text())
print(f"    folds in the log     : {m.group(1) if m else '?'}")
print("    Methods says 'grouped by laboratory, eight folds' with no reason.")
print("    A referee will ask why 8 and not 12, and whether small labs were")
print("    dropped. GroupKFold(n_splits=8) over 12 groups is legal but the")
print("    choice needs a stated reason.")

print("\n  === 6. Abstract says '162 HPLC-assayed samples'. Verify. ===")
z = ROOT / "data/raw/zenodo_13823859"
files = list(z.glob("*"))
print(f"    files present: {[f.name for f in files][:4]}")
log5 = (EV / "phase5_controlled.txt").read_text()
for line in log5.splitlines()[:6]:
    if line.strip():
        print(f"    log: {' '.join(line.split())}")
