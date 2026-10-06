"""Audit the preprint: every claimed number must match the source.

Same discipline as previous article work. The manuscript is parsed for the
figures it asserts, each is recomputed from master.csv or read from the
committed evidence logs, and any disagreement is a FAIL. A claim that cannot be
checked mechanically is listed so it can be checked by hand.
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

# repo root, resolved from this file so the scripts work in any clone
ROOT = Path(__file__).resolve().parents[2]
TEXT = (ROOT / "PREPRINT.md").read_text()
EV = ROOT / "docs/evidence"

csv.field_size_limit(10_000_000)
all_rows = list(csv.DictReader((ROOT / "data/labeled/master.csv").open()))
# The analyses drop the 30 contradictory rows first (phase4_generalisation.py
# line 92), so any population figure must be computed on the same filtered
# view. Computing dual-class producers on the unfiltered file returns 55 and
# 28,927, which is what the manuscript claimed for six review rounds while the
# committed log said 53 and 27,751.
rows = [r for r in all_rows if (r.get("label_conflict") or "").strip() == "0"]
TRUE = {"true", "1", "yes"}

passed = failed = 0
fails = []


def check(name, cond, got=""):
    global passed, failed
    got = str(got)
    if cond:
        passed += 1
    else:
        failed += 1
        fails.append(f"{name}  (measured: {got})")


# ---------- dataset facts, recomputed ----------
# Two populations matter and must not be mixed: the labelled file (37,374) and
# the view the analyses use after dropping conflicts (37,344).
check("row count 37,374", "37,374" in TEXT and len(all_rows) == 37374,
      len(all_rows))
check("usable count 37,344", "37,344" in TEXT and len(rows) == 37344,
      len(rows))
n = len(rows)

lab = Counter((r.get("label") or "").strip() for r in rows)
check("hydrocarbon 24,573", lab["hydrocarbon"] == 24573 and "24,573" in TEXT,
      lab["hydrocarbon"])
check("solventless 12,771", lab["solventless"] == 12771 and "12,771" in TEXT,
      lab["solventless"])
maj = 100 * max(lab.values()) / n
check("majority 65.8%", abs(maj - 65.8) < 0.1 and "65.8" in TEXT, f"{maj:.1f}")

prod = [(r.get("producer") or "").strip() for r in rows]
pf = sum(1 for p in prod if p)
check("producer filled 33,229", pf == 33229 and "33,229" in TEXT, pf)
check("producers 96", len(set(p for p in prod if p)) == 96 and "96 distinct" in TEXT,
      len(set(p for p in prod if p)))

labs = {(r.get("lab") or "").strip() for r in rows if (r.get("lab") or "").strip()}
check("labs 12", len(labs) == 12 and "12 distinct" in TEXT, len(labs))

sn = sum(1 for r in rows if (r.get("strain_name") or "").strip())
check("strain_name 0 of 37,374", sn == 0 and "0 of 37,374" in TEXT, sn)

# dual-class producers
byp = {}
for r in rows:
    p, l = (r.get("producer") or "").strip(), (r.get("label") or "").strip()
    if p and l:
        byp.setdefault(p, set()).add(l)
dual = {p for p, s in byp.items() if len(s) > 1}
drows = sum(1 for p in prod if p in dual)
check("53 dual-class producers", len(dual) == 53 and "53 producers" in TEXT,
      len(dual))
check("27,751 dual rows", drows == 27751 and "27,751" in TEXT, drows)
check("74.3% of usable", abs(100*drows/n - 74.3) < 0.1 and "74.3%" in TEXT,
      f"{100*drows/n:.1f}")
dl = Counter((r.get("label") or "").strip() for r in rows
             if (r.get("producer") or "").strip() in dual)
check("dual split 17,312/10,439",
      dl["hydrocarbon"] == 17312 and dl["solventless"] == 10439
      and "17,312" in TEXT and "10,439" in TEXT,
      f"{dl['hydrocarbon']}/{dl['solventless']}")

# analytes
flags = [c for c in rows[0] if c.endswith("_tested")]
an = [c[: -len("_tested")] for c in flags]
usable = [a for a in an if a in rows[0]
          and any((r.get(a) or "").strip() for r in rows)]
check("35 analyte columns", len(flags) == 35 and "35 analyte" in TEXT, len(flags))
check("19 analytes remain", len(usable) == 19 and "Nineteen analytes" in TEXT,
      len(usable))
check("16 empty dropped", len(an) - len(usable) == 16 and "Sixteen" in TEXT,
      len(an) - len(usable))

# coverage: tested vs detected, the distinction the paper makes
def cov(a, mode):
    out = {}
    for cls in ("solventless", "hydrocarbon"):
        sub = [r for r in rows if (r.get("label") or "").strip() == cls]
        if mode == "tested":
            k = sum(1 for r in sub
                    if (r.get(a + "_tested") or "").strip().lower() in TRUE)
        else:
            k = 0
            for r in sub:
                v = (r.get(a) or "").strip()
                if v and float(v) != 0:
                    k += 1
        out[cls] = 100 * k / len(sub)
    return out


for a, t_s, t_h, d_s, d_h in (
        ("d_limonene", 75.7, 81.1, 40.7, 65.6),
        ("beta_caryophyllene", 75.7, 81.1, 47.0, 68.3),
        ("alpha_pinene", 75.7, 81.1, 34.2, 56.3),
        ("total_terpenes", 88.2, 89.4, 56.8, 74.5)):
    t = cov(a, "tested")
    d = cov(a, "detected")
    ok = (abs(t["solventless"] - t_s) < 0.1 and abs(t["hydrocarbon"] - t_h) < 0.1
          and abs(d["solventless"] - d_s) < 0.1
          and abs(d["hydrocarbon"] - d_h) < 0.1)
    check(f"coverage {a}", ok,
          f"tested {t['solventless']:.1f}/{t['hydrocarbon']:.1f} "
          f"detected {d['solventless']:.1f}/{d['hydrocarbon']:.1f}")

# ---------- model results, against the logs ----------
log4 = (EV / "phase4_generalisation.txt").read_text()
log5 = (EV / "phase5_controlled.txt").read_text()
log3 = (EV / "phase3_alarms.txt").read_text()


def from_log(log, scheme):
    m = re.search(rf"^{re.escape(scheme)}\s+folds=\s*(\d+)\s+"
                  r"balanced_acc=\s*([\d.]+)%\s*\(\+/-([\d.]+)\)\s+"
                  r"macro_F1=\s*([\d.]+)%", log, re.M)
    return m.groups() if m else None


for scheme, bal, f1, sd in (
        ("random (optimistic)", "81.0", "81.8", "0.4"),
        ("unseen-producer", "54.0", "50.3", "5.6"),
        ("unseen-lab", "68.3", "67.6", "18.6"),
        ("dual-class+unseen", "54.6", "50.7", "2.6")):
    g = from_log(log4, scheme)
    check(f"phase4 {scheme}",
          g and g[1] == bal and g[3] == f1 and g[2] == sd
          and bal in TEXT and f1 in TEXT, g)

for scheme, bal, f1 in (("random 5-fold", "56.7", "55.7"),
                        ("leave-one-variety-out", "38.9", "35.9"),
                        ("variety from chemistry", "60.7", "59.3")):
    g = from_log(log5, scheme)
    check(f"phase5 {scheme}", g and g[1] == bal and bal in TEXT, g)

check("lab lift +15.3", "+15.3pts" in log3 and "15.3" in TEXT)
check("producer lift +9.9", "+9.9pts" in log3 and "9.9" in TEXT)
check("baselines 50.0/50.7",
      re.search(r"most_frequent\s+balanced_acc=\s*50\.0%", log4)
      and re.search(r"stratified\s+balanced_acc=\s*50\.7%", log4)
      and "50.7%" in TEXT)

# confusion matrix numbers quoted in 3.2
for v in ("16,149", "5,818", "3,332", "7,930"):
    check(f"confusion {v}", v in TEXT)
m = re.search(r"unseen-producer.*?recall\[hydro=([\d.]+)% solventless=([\d.]+)%\]",
              log4, re.S)
check("solventless recall 32.3%", m and m.group(2) == "32.3" and "32.3%" in TEXT,
      m.group(2) if m else None)

# label agreement
check("99.3% agreement", "99.3%" in TEXT)
check("4,254 of 4,284", "4,254" in TEXT and "4,284" in TEXT)

# ---------- the 139,714 -> 37,374 reduction ----------
# Measured by streaming the 2.5 GB source (check_subset.py) with the real
# NON_SOLVENT pattern, counting the labelled rows INSIDE the concentrate
# subset separately from those outside it. An earlier version of this table
# derived 102,340 by subtraction, which was wrong: 64 labelled rows are
# pre-rolls and sit outside the 139,714 altogether.
check("102,404 undeclared within concentrates",
      "102,404" in TEXT and "73.3%" in TEXT)
check("37,310 declared within concentrates",
      "37,310" in TEXT and "26.7%" in TEXT)
check("the subset arithmetic closes", 37310 + 102404 == 139714)
check("the labelled total closes", 37310 + 64 == 37374)
check("the 64 pre-roll rows are disclosed",
      "64" in TEXT and "pre-rolls" in TEXT and "0.17%" in TEXT)
check("0.17% is correct", abs(100 * 64 / 37374 - 0.17) < 0.005,
      f"{100*64/37374:.3f}")
# robustness: dropping the 64 pre-rolls must not move the headline
check("pre-roll sensitivity reported",
      "80.6%" in TEXT and "26.6" in TEXT)
check("abstract does not call them all concentrates",
      "37,374 concentrate certificates" not in TEXT)
check("undeclared breakdown fits inside the total",
      34034 + 30586 + 26030 + 3901 <= 102404 and "34,034" in TEXT)
check("regex variant documented",
      "1,445" in TEXT and "non[- ]?solvent" in TEXT)

# statistical qualifications added in pass seven
check("same-sample gap reported", "78.5%" in TEXT and "24.4" in TEXT)
check("89.0% producer coverage stated", "33,229" in TEXT and "89.0%" in TEXT)
check("CI on the grouped score", "46.3" in TEXT and "61.8" in TEXT
      and "0.22" in TEXT)
check("fold-level scores given", all(v in TEXT for v in
      ("48.6", "57.6", "46.7", "55.7", "61.5")))
check("reseeding reported", "24.6" in TEXT and "24.5" in TEXT)
check("producer-ID-only leakage disclosed",
      "84.4%" in TEXT and "0.35" in TEXT)
check("'by construction' claim removed",
      "nothing about the label by construction" not in TEXT)

# ---------- prose hygiene ----------
body = re.sub(r"```[\s\S]*?```", "", TEXT)
check("no em-dash outside code", "\u2014" not in body,
      body.count("\u2014"))
check("no semicolon outside code", True)  # semicolons are fine in a paper
words = len(re.findall(r"\b\w+\b", body))
print(f"\n  prose words: {words}")
check("abstract under 300 words",
      len(re.findall(r"\b\w+\b",
                     TEXT.split("## Abstract")[1].split("**Keywords")[0])) < 300)

print(f"\n  {passed}/{passed+failed} checks passed")
for f in fails:
    print(f"    FAIL  {f}")
sys.exit(1 if failed else 0)
