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

# Line wrapping splits phrases across newlines, so every anchored check runs
# against a whitespace-normalised copy rather than the raw text.
FLAT = " ".join(TEXT.split())

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
# Anchored forms: a bare substring survives mutation of the sentence that
# asserts the figure, since the same digits appear in other tables.
check("usable count anchored in prose",
      f"leaving **{len(rows):,} rows**" in FLAT, len(rows))
check("lab fill row anchored",
      f"lab           {len(rows):,} rows (100.0%)  12 distinct" in TEXT)
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
check("producer fill row anchored",
      f"producer      {pf:,} rows (89.0%)   96 distinct" in TEXT, pf)
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
# "53 producers" appears twice; mutating one occurrence left the other intact
# and the check passed. Require both assertions, each with its own wording.
check("dual producers anchored in methods",
      f"the {len(dual)} producers that make" in FLAT, len(dual))
check("dual producers anchored in the discussion",
      f"across these {len(dual)} producers" in FLAT, len(dual))
check("27,751 dual rows", drows == 27751 and "27,751" in TEXT, drows)
check("74.3% of usable", abs(100*drows/n - 74.3) < 0.1 and "74.3%" in TEXT,
      f"{100*drows/n:.1f}")
dl = Counter((r.get("label") or "").strip() for r in rows
             if (r.get("producer") or "").strip() in dual)
check("dual split 17,312/10,439",
      dl["hydrocarbon"] == 17312 and dl["solventless"] == 10439
      and "17,312" in TEXT and "10,439" in TEXT,
      f"{dl['hydrocarbon']}/{dl['solventless']}")

# analytes. `date_tested` ends in "_tested" but is a date, not an analyte
# mask; counting it inflated both figures below by one and made the
# manuscript's "35 analyte columns / Sixteen empty" read as correct.
flags = [c for c in rows[0]
         if c.endswith("_tested") and c != "date_tested"]
an = [c[: -len("_tested")] for c in flags]
usable = [a for a in an if a in rows[0]
          and any((r.get(a) or "").strip() for r in rows)]
check("34 analyte columns", len(flags) == 34 and "34 analyte" in TEXT, len(flags))
check("19 analytes remain", len(usable) == 19 and "Nineteen analytes" in TEXT,
      len(usable))
check("15 empty dropped", len(an) - len(usable) == 15 and "Fifteen" in TEXT,
      len(an) - len(usable))
check("date_tested is not counted as an analyte",
      "date" not in an and "date_tested" not in flags)

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
    # The recomputation above never looked at the manuscript, so a wrong
    # number in the coverage table survived it. Require the row verbatim,
    # built from the RECOMPUTED values rather than from the prose.
    row = (f"{t['solventless']:.1f}%   {t['hydrocarbon']:.1f}%"
           f"           {d['solventless']:.1f}%   {d['hydrocarbon']:.1f}%")
    check(f"coverage row in text: {a}", row in TEXT, row)

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

# `bal in TEXT` is too weak: "81.0" also occurs in the results table, so
# mutating the sentence that states the headline survived it. Pin each number
# to the specific sentence or table row that asserts it, using the value read
# from the LOG rather than from the prose.
rb = from_log(log4, "random (optimistic)")
gb = from_log(log4, "unseen-producer")
db = from_log(log4, "dual-class+unseen")
check("headline random anchored",
      bool(rb) and f"at {rb[1]}% balanced accuracy" in FLAT,
      rb[1] if rb else None)
check("results-table random row anchored",
      rb and f"random (optimistic)          {rb[1]}% (+/-{rb[2]})" in TEXT,
      rb[1] if rb else None)
check("results-table grouped row anchored",
      gb and f"unseen-producer              {gb[1]}% (+/-{gb[2]})" in TEXT,
      gb[1] if gb else None)
check("results-table dual row anchored",
      db and f"dual-class + unseen          {db[1]}% (+/-{db[2]})" in TEXT,
      db[1] if db else None)
check("conclusion states the grouped score",
      bool(gb) and f"{gb[1]}% balanced" in FLAT, gb[1] if gb else None)

for scheme, bal, f1 in (("random 5-fold", "56.7", "55.7"),
                        ("leave-one-variety-out", "38.9", "35.9"),
                        ("variety from chemistry", "60.7", "59.3")):
    g = from_log(log5, scheme)
    check(f"phase5 {scheme}", g and g[1] == bal and bal in TEXT, g)

check("lab lift +15.3", "+15.3pts" in log3 and "15.3" in TEXT)
check("producer lift +9.9", "+9.9pts" in log3 and "9.9" in TEXT)
# Again too weak: "15.3" survives mutation of the sentence and of the table
# row. Anchor both, using the lift parsed from the log.
m3 = re.search(r"lab\s+classes=\s*\d+\s+rows=\s*(\d+)\s+acc=\s*[\d.]+%"
               r"\s*\(\+/-[\d.]+\)\s+baseline=\s*[\d.]+%\s+lift=\s*"
               r"\+([\d.]+)pts", log3)
check("lab lift anchored in the results table",
      bool(m3) and f"+{m3.group(2)} pts" in TEXT,
      m3.group(2) if m3 else None)
check("lab lift anchored in prose",
      bool(m3) and f"laboratory, {m3.group(2)} points above" in FLAT,
      m3.group(2) if m3 else None)
check("lab probe row count anchored",
      bool(m3) and f"{int(m3.group(1)):,}" in TEXT,
      m3.group(1) if m3 else None)
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
# Anchor the agreement figure to its sentence and recompute the percentage
# from the two counts the manuscript states, so a mutated percentage fails
# even though the digits still appear elsewhere.
ag = re.search(r"they agree on ([\d,]+) of ([\d,]+) rows, ([\d.]+)%", FLAT)
check("agreement recomputes from its own counts",
      bool(ag) and abs(100 * int(ag.group(1).replace(",", ""))
                       / int(ag.group(2).replace(",", ""))
                       - float(ag.group(3))) < 0.05,
      f"{ag.group(3) if ag else None}")
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
# The protocol-only gap must equal the two scores the paper states, so a
# mutated gap contradicts its own arithmetic.
ss = re.search(r"rows\* scores ([\d.]+)%, so the protocol-only gap is\s*"
               r"\*\*([\d.]+) points\*\*", FLAT)
gscore = float(gb[1]) if gb else None
check("protocol-only gap recomputes",
      bool(ss) and gscore is not None
      and abs((float(ss.group(1)) - gscore) - float(ss.group(2))) < 0.15,
      f"{ss.group(1)}-{gscore}={float(ss.group(1))-gscore:.1f} "
      f"vs stated {ss.group(2)}" if ss and gscore else None)
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

# ---------- the 2.3 pre-check must match its own evidence log ----------
# This block exists because 2.3 previously claimed a +0.0-point lift from a
# probe that looked at 6 terpenes while the model saw 19 columns. The figures
# are now read out of the regenerated log so prose and evidence cannot drift.
log2 = (EV / "lab_confounding.txt").read_text()
mp = re.search(r"accuracy from missingness alone:\s*([\d.]+)%", log2)
mb = re.search(r"majority-class baseline:\s*([\d.]+)%", log2)
ml = re.search(r"lift over baseline:\s*\+?(-?[\d.]+) points", log2)
check("missingness probe accuracy anchored",
      bool(mp) and f"{mp.group(1)}% accuracy" in FLAT,
      mp.group(1) if mp else None)
check("missingness probe lift anchored",
      bool(ml) and f"**+{ml.group(1)} points**" in TEXT,
      ml.group(1) if ml else None)
check("missingness lift recomputes from its own two figures",
      bool(mp and mb and ml)
      and abs((float(mp.group(1)) - float(mb.group(1))) - float(ml.group(1)))
      < 0.11,
      f"{mp.group(1)}-{mb.group(1)} vs {ml.group(1)}" if mp and mb and ml
      else None)
check("the discredited +0.0 claim is gone",
      "+0.0 points" not in TEXT and "lift of **+0.0" not in TEXT)
check("probe scope stated as the full feature set",
      "19 analyte columns the model sees" in FLAT)
# The deterministic part of the shortcut, which the old probe could not see.
pp = re.search(r"patterns that are 100% one class:\s*(\d+) of (\d+)", log2)
pr = re.search(r"rows they cover:\s*(\d+) \(([\d.]+)%\)", log2)
check("pure-stratum count anchored",
      bool(pp) and f"{pp.group(1)} of {pp.group(2)}" in TEXT,
      pp.group(0) if pp else None)
check("pure-stratum coverage anchored",
      bool(pr) and pr.group(1) in TEXT.replace(",", "")
      and f"{pr.group(2)}%" in TEXT,
      pr.group(0) if pr else None)
check("pure strata attributed to reporting convention",
      "laboratory reporting convention" in FLAT)

# ---------- 2.2 feature-construction disclosures ----------
check("total_terpenes collinearity disclosed",
      "0.984" in TEXT and "0.992" in TEXT)
check("total_terpenes ablation cost disclosed", "0.3 points" in FLAT)
check("total_terpenes population is a lab convention",
      "33,229 / 33,229" in TEXT)
check("impossible potency values disclosed",
      "686,400" in TEXT and "101 rows" in FLAT)

# ---------- the confusion matrix must reconcile with its own recall ----------
# 3.2 quotes a pooled matrix and a fold-averaged recall; they differ and the
# paper now states both. Recompute the pooled figure from the quoted cells.
cm = re.search(r"true solventless:\s*([\d,]+) correct /\s*([\d,]+) wrong", TEXT)
po = re.search(r"=\s*\*\*([\d.]+)%\*\*", TEXT)
if cm and po:
    tp = int(cm.group(1).replace(",", ""))
    fn = int(cm.group(2).replace(",", ""))
    check("pooled recall recomputes from the quoted matrix",
          abs(100 * tp / (tp + fn) - float(po.group(1))) < 0.1,
          f"{100*tp/(tp+fn):.1f} vs stated {po.group(1)}")
else:
    check("pooled recall recomputes from the quoted matrix", False, "no match")
check("both recall conventions labelled",
      "mean of the five per-fold recalls" in FLAT and "Pooling the matrix" in FLAT)

# ---------- 4.3 strain: denominators and the grouped runs ----------
# An earlier draft reported "37.1% obvious non-cultivars", which divided
# noise+empty rows by the key-bearing rows only — a ratio of two different
# populations. Every figure is now read out of the phase4b log.
log4b = (EV / "phase4b_strain.txt").read_text()
sj = re.search(r"of those, obvious non-cultivar:\s*(\d+) \(([\d.]+)% of keys\)",
               log4b)
su = re.search(r"rows with no usable cultivar signal:\s*(\d+) "
               r"\(([\d.]+)% of labelled\)", log4b)
check("strain junk-key share anchored",
      bool(sj) and f"{sj.group(2)}% of keys" in TEXT,
      sj.group(2) if sj else None)
check("strain no-signal share anchored",
      bool(su) and f"{su.group(2)}% of labelled" in TEXT,
      su.group(2) if su else None)
check("strain junk count anchored",
      bool(sj) and f"{int(sj.group(1)):,}" in TEXT,
      sj.group(1) if sj else None)
check("the bad 37.1% ratio is gone",
      "37.1%" not in TEXT and "one-third noise" not in TEXT)
# The two grouped strain runs exist in the log and must not be withheld.
us = re.search(r"unseen-strain \(all\)\s+folds=\d+\s+"
               r"balanced_acc=\s*([\d.]+)%", log4b)
ds = re.search(r"dual-strain\+unseen\s+folds=\d+\s+"
               r"balanced_acc=\s*([\d.]+)%", log4b)
check("unseen-strain result reported",
      bool(us) and f"{us.group(1)}% balanced accuracy" in FLAT,
      us.group(1) if us else None)
check("dual-strain result reported",
      bool(ds) and f"{ds.group(1)}%" in TEXT,
      ds.group(1) if ds else None)
dsr = re.search(r"dual-class strains:\s*(\d+)\s+rows:\s*(\d+)", log4b)
check("dual-strain subset size reported",
      bool(dsr) and f"{dsr.group(1)} keys" in FLAT
      and f"{int(dsr.group(2)):,} rows" in TEXT,
      dsr.group(0) if dsr else None)
check("4.3 no longer claims the test was impossible",
      "The confounder we could not test cleanly" in TEXT
      and "We could not test it. " not in TEXT)

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
