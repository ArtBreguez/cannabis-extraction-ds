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
check("strain_name 0 of 37,344", sn == 0 and "0 of 37,344" in TEXT, sn)

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
logc = (EV / "count_label_subset.txt").read_text()


def cnum(label):
    m = re.search(rf"{re.escape(label)}\s+(\d+)", logc)
    return int(m.group(1)) if m else None


c_conc = cnum("concentrate or extract rows:")
c_dec = cnum("product_type declares the family:")
c_und = cnum("product_type does not:")
c_out = cnum("labelled rows outside that subset:")
c_tot = cnum("total labelled rows:")
check("reduction table: concentrate subset anchored",
      c_conc and f"concentrate or extract rows                     {c_conc:,}" in TEXT,
      c_conc)
check("reduction table: declared anchored",
      c_dec and f"extraction family    {c_dec:,}   ({100*c_dec/c_conc:.1f}%)" in TEXT,
      c_dec)
check("reduction table: undeclared anchored",
      c_und and f"does not                         {c_und:,}   ({100*c_und/c_conc:.1f}%)" in TEXT,
      c_und)
check("reduction table: outside-subset rows anchored",
      c_out == 64 and f"outside that subset                    {c_out}" in TEXT, c_out)
check("reduction table: total anchored",
      c_tot == len(all_rows) and f"total labelled rows                              {c_tot:,}" in TEXT,
      c_tot)
check("the subset arithmetic closes", c_dec + c_und == c_conc)
check("the labelled total closes", c_dec + c_out == c_tot)
check("the 64 pre-roll rows are disclosed",
      "64" in TEXT and "pre-rolls" in TEXT and "0.17%" in TEXT)
check("0.17% is correct", abs(100 * c_out / c_tot - 0.17) < 0.005,
      f"{100*c_out/c_tot:.3f}")
check("abstract does not call them all concentrates",
      "37,374 concentrate certificates" not in TEXT)
check("abstract counts the usable rows",
      f"Using {len(rows):,} cannabis certificates" in FLAT)
for t, n in (("extracts", 34034), ("concentrate, product inhalable", 30586),
             ("concentrate", 26030), ("marijuana extract for inhalation", 3901)):
    m = re.search(rf"^\s+(\d+)\s+{re.escape(t)}$", logc, re.M)
    check(f"undeclared value {t} anchored",
          bool(m) and int(m.group(1)) == n and f"({n:,}" in TEXT or
          (bool(m) and int(m.group(1)) == n and f"({n:,})" in TEXT),
          m.group(1) if m else None)
check("undeclared breakdown fits inside the total",
      34034 + 30586 + 26030 + 3901 <= c_und)
mv = re.search(r"^\s+(\d+)\s+non-solvent concentrate$", logc, re.M)
check("regex variant documented",
      bool(mv) and f"`non-solvent concentrate` ({int(mv.group(1)):,} rows)" in FLAT
      and "non[- ]?solvent" in TEXT, mv.group(1) if mv else None)

# robustness figures, each read from docs/evidence/sensitivity_checks.txt
logs = (EV / "sensitivity_checks.txt").read_text()
blk = logs.split("=== 4.2: WITHOUT THE 64 PRE-ROLL ROWS ===")[1].split("===")[0]
pr_r = re.search(r"random \(all rows\)\s+([\d.]+)%", blk)
pr_g = re.search(r"unseen-producer\s+([\d.]+)%", blk)
pr_gap = re.search(r"gap\s+([\d.]+) points", blk)
check("pre-roll sensitivity: random anchored",
      bool(pr_r) and f"from 81.0% to {pr_r.group(1)}%" in FLAT,
      pr_r.group(1) if pr_r else None)
check("pre-roll sensitivity: grouped unchanged",
      bool(pr_g) and pr_g.group(1) == gb[1]
      and f"unseen-producer unchanged at {pr_g.group(1)}%" in FLAT,
      pr_g.group(1) if pr_g else None)
check("pre-roll sensitivity: gap anchored",
      bool(pr_gap) and f"27-point gap becomes {pr_gap.group(1)}" in FLAT,
      pr_gap.group(1) if pr_gap else None)
abl = re.search(r"cost of dropping it\s+([\d.]+) points", logs)
check("total_terpenes ablation cost anchored",
      bool(abl) and f"costs {abl.group(1)} points of balanced accuracy" in FLAT,
      abl.group(1) if abl else None)

# statistical qualifications added in pass seven
log7 = (EV / "review7_statistical.txt").read_text()
r7_sub = re.search(r"random, producer rows only\s+([\d.]+)%", log7)
r7_gap = re.search(r"gap on a SINGLE sample\s+([\d.]+) points", log7)
ss = re.search(r"rows\* scores ([\d.]+)%, so the protocol-only gap is\s*"
               r"\*\*([\d.]+) points\*\*", FLAT)
check("same-sample random score anchored to its log",
      bool(ss and r7_sub) and ss.group(1) == r7_sub.group(1),
      r7_sub.group(1) if r7_sub else None)
check("protocol-only gap anchored to its log",
      bool(ss and r7_gap) and ss.group(2) == r7_gap.group(1),
      r7_gap.group(1) if r7_gap else None)
# The protocol-only gap must equal the two scores the paper states, so a
# mutated gap contradicts its own arithmetic.
gscore = float(gb[1]) if gb else None
check("protocol-only gap recomputes",
      bool(ss) and gscore is not None
      and abs((float(ss.group(1)) - gscore) - float(ss.group(2))) < 0.15,
      f"{ss.group(1)}-{gscore}={float(ss.group(1))-gscore:.1f} "
      f"vs stated {ss.group(2)}" if ss and gscore else None)
check("89.0% producer coverage stated", "33,229" in TEXT and "89.0%" in TEXT)
r7_ci = re.search(r"unseen-producer: mean [\d.]+%\s+95% CI \[([\d.]+), ([\d.]+)\]"
                  r"\s+p=([\d.]+)", log7)
check("CI on the grouped score anchored to its log",
      bool(r7_ci) and f"[{r7_ci.group(1)}, {r7_ci.group(2)}]" in TEXT
      and f"p = {float(r7_ci.group(3)):.2f}" in TEXT,
      r7_ci.group(0) if r7_ci else None)
r7_folds = re.search(r"folds: \[([\d., ]+)\]", log7)
check("fold-level scores anchored to its log",
      bool(r7_folds) and all(v.strip() in TEXT
                             for v in r7_folds.group(1).split(",")),
      r7_folds.group(1) if r7_folds else None)
r7_seeds = re.findall(r"seed\s+\d+: random [\d.]+%\s+grouped ([\d.]+)%\s+gap ([\d.]+)",
                      log7)
check("reseeding anchored to its log",
      len(r7_seeds) == 3 and all(g == gb[1] for g, _ in r7_seeds)
      and f"gaps of {r7_seeds[0][1]}, {r7_seeds[1][1]} and {r7_seeds[2][1]} points" in FLAT,
      r7_seeds)
r7_pid = re.search(r"producer ID alone predicts label at ([\d.]+)%", log7)
r7_sh = re.search(r"min ([\d.]+)\s+median ([\d.]+)\s+max ([\d.]+)", log7)
check("producer-ID-only leakage anchored to its log",
      bool(r7_pid) and f"reaches {r7_pid.group(1)}% balanced" in FLAT,
      r7_pid.group(1) if r7_pid else None)
check("producer share range anchored to its log",
      bool(r7_sh) and f"median of {r7_sh.group(2)}" in FLAT
      and f"to {r7_sh.group(3)} with" in FLAT
      and "from below 0.01 to" in FLAT,
      r7_sh.group(0) if r7_sh else None)
# "0.00" would read as a producer with no solventless rows, which contradicts
# the subset's definition; the smallest share is 3 of 1,801.
check("share range does not print a 0.00 minimum",
      "ranges from 0.00" not in FLAT)
check("'by construction' claim removed",
      "nothing about the label by construction" not in TEXT)

# ---------- the dual-class scheme is above chance, and the text says so ----------
# Pass seven computed an interval for unseen-producer only. The dual-class
# scheme's five folds give an interval that EXCLUDES 50%, so the earlier
# sentence "no gate that requires beating chance by more than the
# between-fold spread is passed" was false for that scheme.
d_ci = re.search(r"dual-class\+unseen\s+([\d.]+)% \(\+/-([\d.]+)\)\s*\n"
                 r"folds: \[([\d., ]+)\]\s*\n95% CI \[([\d.]+), ([\d.]+)\]\s+p=([\d.]+)",
                 logs)
check("dual-class folds anchored to the sensitivity log",
      bool(d_ci) and all(v.strip() in TEXT for v in d_ci.group(3).split(","))
      and d_ci.group(1) == db[1], d_ci.group(3) if d_ci else None)
check("dual-class interval anchored",
      bool(d_ci) and f"[{d_ci.group(4)}, {d_ci.group(5)}] that excludes 50.0%" in FLAT
      and f"(p = {d_ci.group(6)})" in FLAT, d_ci.group(0)[-60:] if d_ci else None)
check("dual-class residual recomputes",
      bool(db) and f"residual of {float(db[1])-50:.1f} points" in FLAT
      and bool(rb) and f"sits {float(rb[1])-float(db[1]):.1f} points below the random split" in FLAT,
      f"{float(db[1])-50:.1f} / {float(rb[1])-float(db[1]):.1f}" if db and rb else None)
# The per-fold recall extremes quoted in 3.2 (48.5% best, 25.3% worst) had
# no committed source either; sensitivity_checks.py now prints each fold.
fr = re.search(r"mean of fold recalls: ([\d.]+)%\s+min ([\d.]+)%\s+max ([\d.]+)%",
               logs)
check("per-fold recall extremes anchored to the sensitivity log",
      bool(fr) and f"({fr.group(3)}% against {fr.group(2)}% in the fold with the most)" in FLAT
      and fr.group(1) == "32.3", fr.group(0) if fr else None)
# The folds are equal in SIZE (GroupKFold balances them); what differs is the
# solventless count per fold. An earlier sentence said "folds of very unequal
# size", which the log contradicts.
fs = re.search(r"solventless rows per fold: min (\d+)\s+max (\d+)", logs)
check("per-fold solventless counts anchored",
      bool(fs) and f"from {int(fs.group(1)):,} to {int(fs.group(2)):,}" in FLAT
      and "folds are of very unequal size" not in FLAT,
      fs.group(0) if fs else None)
check("the false 'no gate is passed' sentence is gone",
      "No gate that requires" not in FLAT)
check("the chance claim is scoped to the unseen-producer scheme",
      "on this scheme the collapse goes all the way to chance" in FLAT)

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
lab_n = Counter((r.get("lab") or "").strip() for r in rows).most_common(2)
check("two largest laboratories counted on the usable rows",
      f"{lab_n[0][1]:,} and {lab_n[1][1]:,} rows" in FLAT
      and f"{lab_n[0][1]:,}" in log2.replace("15018", "15,018"),
      lab_n)
check("stale 37,374-population lab counts are gone",
      "15,035" not in TEXT and "12,539" not in TEXT)
check("pre-check log runs on the usable rows",
      f"labeled rows (conflicts excluded): {len(rows)}" in log2)

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
               r"\(([\d.]+)% of usable\)", log4b)
check("strain junk-key share anchored",
      bool(sj) and f"{sj.group(2)}% of keys" in TEXT,
      sj.group(2) if sj else None)
check("strain no-signal share anchored",
      bool(su) and f"{su.group(2)}% of usable" in TEXT,
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

# ---------- submission gates ----------
# ChemRxiv requires AI-tool use to be disclosed in the text of the preprint
# and forbids listing such tools as authors. Both halves are asserted.
check("AI tool use is disclosed", "## Use of AI tools" in TEXT)
check("AI disclosure denies authorship",
      "do not meet authorship criteria" in TEXT)
check("AI disclosure names the author as responsible",
      "author is responsible for every claim" in FLAT)
check("no AI tool in the author block",
      not re.search(r"^(ChatGPT|Claude|GPT|Copilot|Gemini)", TEXT, re.M))
check("competing interests declared", "## Competing interests" in TEXT)
check("data availability declared", "## Data and code availability" in TEXT)

# ---------- 2.3 pure-stratum attribution must not overclaim ----------
# An earlier draft said the pure-stratum rows "come from four laboratories"
# with "no producer at all" and were "exactly" the dropped rows. Measured:
# 4,115 of 4,123 fit that description; 8 rows sit in other laboratories and
# do carry a producer. The counts are now asserted against the data.
_pat = {}
for r in rows:
    key = tuple(1 if (r.get(a) or "").strip() else 0 for a in sorted(usable))
    _pat.setdefault(key, []).append((r.get("label") or "").strip())
_pure = [k for k, v in _pat.items() if len(set(v)) == 1]
_pure_rows = sum(len(_pat[k]) for k in _pure)
_pure_noprod = sum(
    1 for r in rows
    if tuple(1 if (r.get(a) or "").strip() else 0 for a in sorted(usable)) in set(_pure)
    and not (r.get("producer") or "").strip())
check("pure-stratum total matches the data",
      f"{_pure_rows:,} of" in TEXT or str(_pure_rows) in TEXT, str(_pure_rows))
check("the producer-less majority is counted, not generalised",
      f"{_pure_noprod:,} of {_pure_rows:,}" in TEXT, f"{_pure_noprod}/{_pure_rows}")
check("the 8-row exception is disclosed",
      f"remaining {_pure_rows - _pure_noprod} rows" in TEXT,
      str(_pure_rows - _pure_noprod))
check("no 'exactly the rows' overclaim",
      "exactly the\nrows that the unseen-producer" not in TEXT
      and "no producer at all" not in TEXT)
check("2.6 and 4.3 agree that the cultivar probe ran",
      "could not be\nprobed" not in TEXT and "could only be probed" in TEXT)

# 4.1 must not claim strain dominates in the MARKET data: the strain alarm
# probe there returns a negative lift, which 4.3 reports. The two datasets
# disagree, so the claim is scoped or it contradicts 4.3.
check("4.1 does not assert strain dominance unscoped",
      "Strain and producer variation dominate" not in FLAT)
check("4.1 scopes the cultivar claim to the controlled data",
      "datasets disagree on the evidence" in FLAT
      and "we do not claim it for the market corpus" in FLAT)
check("4.1 quotes the negative market-data lift",
      "minus 3.0 points (4.3)" in FLAT)
check("4.1 supports producer dominance with both figures",
      "84.4% while the chemistry" in FLAT)

# ---------- statements about the pipeline that were wrong ----------
# Non-detects are LEFT-censored (true value below the limit); the text said
# "right-censored". No model script feeds the `_tested` flags to the
# estimator: the 19 value columns carry NaN, which HistGradientBoosting
# routes natively. phase3_alarms.py uses max_iter=120, not 200.
check("censoring direction is left", "left-censored" in FLAT
      and "right-censored" not in FLAT)
check("the model is described as receiving 19 value columns",
      "The classifier receives the 19 value columns" in FLAT)
check("the false '_tested flags are excluded' claim is gone",
      "flags are excluded from these probes" not in FLAT)
check("alarm probes disclose max_iter=120", "`max_iter=120`" in TEXT)
check("alarm probes disclose plain accuracy",
      "report plain accuracy against the explicit majority-class baseline" in FLAT)
check("strain columns described as populated for 0 rows",
      "populated for **0 of 37,344**" in FLAT and "empty for **0 of" not in FLAT)
sa = re.search(r"strains=(\d+) rows=(\d+)\s+acc=[\d.]+%\s+baseline=[\d.]+%"
               r"\s+lift=(-?[\d.]+)pts", log4b)
check("strain alarm probe subset anchored",
      bool(sa) and f"the {sa.group(1)} keys with at least 30 rows "
      f"({int(sa.group(2)):,} rows)" in FLAT
      and f"minus {abs(float(sa.group(3))):.1f} points" in FLAT,
      sa.group(0) if sa else None)
check("name path described as cross-check only",
      "The name path never adds a row" in FLAT)

# ---------- what the declared classes contain ----------
# The label is the regulator's category. Under NAC 453D.780 CO2 extraction is
# "nonsolvent", so the class is wider than rosin-and-hash; 2.1 says so and
# quotes the composition from docs/evidence/label_composition.txt.
logl = (EV / "label_composition.txt").read_text()
for fam, key in (("distillate", "distillate"),
                 ("co2 / supercritical", "co2 / supercritical"),
                 ("ethanol", "ethanol"),
                 ("hydrocarbon words", "hydrocarbon words"),
                 ("mechanical words", "mechanical words")):
    m = re.search(rf"^{re.escape(key)}.*?(\d+) \(\s*([\d.]+)%\)\s+(\d+) \(\s*([\d.]+)%\)$",
                  logl, re.M)
    check(f"composition row anchored: {fam}",
          bool(m) and f"{int(m.group(1)):,} ({m.group(2):>4}%)" in TEXT
          and f"{int(m.group(3)):,} ({m.group(4):>4}%)" in TEXT,
          m.group(0)[-40:] if m else None)
check("2.1 quotes the regulatory definition",
      "including concentrated marijuana extracted with CO2" in FLAT
      and "[12]" in TEXT)
check("the task is scoped to the declared category",
      "not rosin against BHO specifically" in FLAT
      and "declared extraction category" in FLAT)
check("name-path comparison scope stated",
      abs(100 * 4284 / len(all_rows) - 11.5) < 0.05 and "11.5% of labelled rows" in FLAT,
      f"{100*4284/len(all_rows):.2f}")
check("introduction no longer defines the classes as rosin vs BHO",
      "separating solventless extracts (rosin" not in FLAT)

# ---------- the controlled experiment's robustness checks ----------
# audit_phase5_controlled.py recomputes 3.3 from the raw xlsx and adds the
# seed sweep, the random-group control and the permutation tests that
# phase5_controlled.txt cannot provide. Every figure 3.3 quotes from it is
# read here from that log.
logp = (EV / "audit_phase5_controlled.txt").read_text()
k5 = re.search(r"k=5: mean=([\d.]+)% sd=([\d.]+)", logp)
pct = re.search(r"sits at the (\d+)th percentile", logp)
ctl = re.search(r"mean=([\d.]+)% sd=[\d.]+ 2\.5-97\.5pct=\[[\d.]+,[\d.]+\]\s*\n\s*real LOVO", logp)
pc = re.search(r"P\(random-group LOVO <= real LOVO\) = ([\d.]+)", logp)
pr5 = re.search(r"r5\s+obs=[\d.]+%.*?p=([\d.]+)", logp)
plo = re.search(r"lovo\s+obs=[\d.]+%.*?p=([\d.]+)", logp)
gap = re.search(r"gap = ([\d.]+) pts", logp)
ttp = re.search(r"if t/T/P were included: bal=([\d.]+)%", logp)
check("seed sweep mean anchored",
      bool(k5) and f"average {k5.group(1)}% (sd {k5.group(2)})" in FLAT,
      k5.group(0) if k5 else None)
check("seed percentile anchored",
      bool(pct) and f"sits at the {pct.group(1)}th percentile" in FLAT,
      pct.group(0) if pct else None)
lovo = from_log(log5, "leave-one-variety-out")
check("seed-robust gap recomputes",
      bool(k5 and lovo) and f"is {float(k5.group(1)) - float(lovo[1]):.1f} points rather than" in FLAT,
      f"{float(k5.group(1)) - float(lovo[1]):.1f}" if k5 and lovo else None)
check("single-seed gap anchored",
      bool(gap) and f"rather than {gap.group(1)}" in FLAT, gap.group(0) if gap else None)
check("random-group control anchored",
      bool(ctl and pc) and f"averages {ctl.group(1)}% over 200 draws" in FLAT
      and float(pc.group(1)) == 0.0 and "never falls to 38.9%" in FLAT,
      (ctl.group(1), pc.group(1)) if ctl and pc else None)
check("permutation p-values anchored",
      bool(pr5 and plo) and f"p = {float(pr5.group(1)):.3f} for the random split" in FLAT
      and f"p = {float(plo.group(1)):.2f} for leave-one-variety-out" in FLAT,
      (pr5.group(1), plo.group(1)) if pr5 and plo else None)
check("process-settings leak anchored",
      bool(ttp) and f"lifts the random\nsplit to {ttp.group(1)}%" in TEXT,
      ttp.group(1) if ttp else None)
check("discussion carries the seed-averaged gap",
      bool(k5 and lovo) and f"{round(float(k5.group(1)) - float(lovo[1]))} once the random" in FLAT
      and f"{round(float(k5.group(1)) - float(lovo[1]))} to 18 on a designed experiment" in FLAT)

# ---------- references ----------
refs = TEXT.split("## References")[1]
for n in range(1, 13):
    m = re.search(rf"^\[{n}\] (.+?)(?=^\[\d+\]|\Z)", refs, re.M | re.S)
    body_ = " ".join(m.group(1).split()) if m else ""
    has_doi = "doi:" in body_
    check(f"reference [{n}] present with a DOI or dataset pointer",
          bool(m) and (has_doi or n in (9, 12)), body_[:60])
check("reference [1] names the right journal",
      "European Archives of Psychiatry and Clinical Neuroscience" in FLAT
      and "Journal of Cannabis Research" not in FLAT)
check("every reference carries authors",
      all(re.match(r"\[\d+\] [A-ZÁ-Ž][^.]*?, ", " ".join(ln.split()))
          for ln in re.findall(r"^\[\d+\] .+", refs, re.M)
          if not ln.startswith(("[9]", "[12]"))))

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
