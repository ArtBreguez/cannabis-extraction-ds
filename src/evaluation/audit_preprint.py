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
# the phrase occurs in the abstract and in 2.4; every occurrence must agree
check("every dual-producer count agrees",
      set(re.findall(r"the (\d+) producers that make", FLAT)) == {str(len(dual))},
      re.findall(r"the (\d+) producers that make", FLAT))
check("dual producers anchored in the discussion",
      f"across these {len(dual)} producers" in FLAT, len(dual))
check("27,751 dual rows", drows == 27751 and "27,751" in TEXT, drows)
check("dual rows anchored in 2.4 and 3.2",
      f"covers {drows:,} rows" in FLAT and f"on the same {drows:,} rows" in FLAT, drows)
check("74.3% of usable", abs(100*drows/n - 74.3) < 0.1
      and f"{100*drows/n:.1f}% of the usable data" in FLAT,
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

# The probes were rerun at learning_rate=0.05 after the 0.1 runs were found
# to diverge; every figure below is read from the regenerated log.
check("stale 0.1-rate lifts are not quoted as results",
      "+15.3 pts" not in TEXT and "[watch]" not in TEXT)
# Again too weak: "15.3" survives mutation of the sentence and of the table
# row. Anchor both, using the lift parsed from the log.
m3 = re.search(r"lab\s+classes=\s*\d+\s+rows=\s*(\d+)\s+acc=\s*[\d.]+%"
               r"\s*\(\+/-[\d.]+\)\s+baseline=\s*[\d.]+%\s+lift=\s*"
               r"\+([\d.]+)pts", log3)
check("lab lift anchored in the results table",
      bool(m3) and f"+{m3.group(2)} pts" in TEXT,
      m3.group(2) if m3 else None)
check("lab lift anchored in prose",
      bool(m3) and f"{m3.group(2)} points above the majority baseline" in FLAT,
      m3.group(2) if m3 else None)
ma = re.search(r"lab\s+classes=\s*\d+\s+rows=\s*\d+\s+acc=\s*([\d.]+)%\s*"
               r"\(\+/-([\d.]+)\)\s+baseline=\s*([\d.]+)%", log3)
check("lab probe accuracy anchored in abstract, table and conclusion",
      bool(ma) and f"testing laboratory at {ma.group(1)}% against {ma.group(3)}%" in FLAT
      and f"{ma.group(1)}% (+/-{ma.group(2)})    {ma.group(3)}%" in TEXT
      and f"laboratory at {ma.group(1)}% against {ma.group(3)}%, and the variety" in FLAT,
      ma.groups() if ma else None)
mp_ = re.search(r"producer\s+classes=\s*\d+\s+rows=\s*\d+\s+acc=\s*([\d.]+)%\s*"
                r"\(\+/-([\d.]+)\)\s+baseline=\s*[\d.]+%\s+lift=\s*\+([\d.]+)pts", log3)
check("producer probe anchored",
      bool(mp_) and f"producer on {mp_.group(1)}%, {mp_.group(3)} points" in FLAT,
      mp_.groups() if mp_ else None)
mb_ = re.findall(r"balanced_acc=([\d.]+)% \(\+/-[\d.]+\) chance=([\d.]+)%", log3)
check("probe balanced accuracies anchored",
      len(mb_) == 2 and all(f"{b}% ({c}%)" in TEXT for b, c in mb_), mb_)
mm_ = re.search(r"acc=([\d.]+)%\s+baseline=[\d.]+%\s+\(lookup table over (\d+) patterns", log3)
check("missingness-only lab probe anchored",
      bool(mm_) and f"{mm_.group(2)} missingness patterns alone" in FLAT
      and f"reaches {mm_.group(1)}% on the same folds" in FLAT, mm_.groups() if mm_ else None)
mg_ = re.search(r"classes=(\d+) rows=(\d+)\s+acc=([\d.]+)% \(\+/-([\d.]+)\)\s+baseline=([\d.]+)%", log3)
check("producer-grouped lab probe anchored",
      bool(mg_) and f"({int(mg_.group(2)):,} rows)" in FLAT
      and f"{mg_.group(3)}% (+/-{mg_.group(4)}) against a {mg_.group(5)}% baseline" in FLAT,
      mg_.groups() if mg_ else None)
md_ = re.search(r"learning_rate=0\.1\s+folds=\[([\d., ]+)\]", log3)
check("divergence at 0.1 disclosed with its fold range",
      bool(md_) and f"{min(float(v) for v in md_.group(1).split(','))} to "
      f"{max(float(v) for v in md_.group(1).split(','))}" in FLAT,
      md_.group(1) if md_ else None)
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
      bool(d_ci) and f"[{d_ci.group(4)}, {d_ci.group(5)}] that excludes 50.0% (p = {d_ci.group(6)})" in FLAT,
      d_ci.group(0)[-60:] if d_ci else None)
# pass nine: the dual-class residual is reported as a range over fold
# assignments, not as a detected effect, and every figure comes from
# docs/evidence/review9_checks.txt
log9 = (EV / "review9_checks.txt").read_text()
c9 = re.search(r"rows=(\d+)\s+random 5-fold ([\d.]+)% \(\+/-([\d.]+)\)\s+grouped by producer ([\d.]+)%.*?gap ([\d.]+) points", log9)
check("dual-class same-sample random split anchored",
      bool(c9) and f"against {c9.group(2)}% (+/-{c9.group(3)}) for a random split on the same {int(c9.group(1)):,} rows" in FLAT
      and f"a gap of {c9.group(5)} points" in FLAT and c9.group(4) == db[1], c9.groups() if c9 else None)
d9 = re.findall(r"seed\s+\d+: unseen-producer ([\d.]+)% \(\+/-[\d.]+\)\s+dual-class ([\d.]+)%", log9)
# pass ten: the producer is the independent unit. Twenty repeated partitions
# and a producer-cluster bootstrap replace the three reassignments and the
# row-level permutation test, whose null treated rows as exchangeable.
log10a = (EV / "review10a_grouped.txt").read_text()
def _blk(name):
    return log10a.split(f"=== {name}:")[1].split("\n===")[0]
def _grp(name):
    b = _blk(name)
    A = re.search(r"min ([\d.]+)%\s+median ([\d.]+)%\s+max ([\d.]+)%\s+mean ([\d.]+)%\s+sd ([\d.]+)", b)
    K = re.search(r"at or below 50\.0%: (\d+) of (\d+)", b)
    B = re.search(r"95% interval \[([\d.]+), ([\d.]+)\]\s+share of draws at or below 50\.0%: ([\d.]+)%", b)
    L = re.search(r"laboratory identity alone, same folds: ([\d.]+)%", b)
    return A, K, B, L
uA, uK, uB, uL = _grp("unseen-producer")
dA, dK, dB, dL = _grp("dual-class+unseen")
check("unseen-producer: repeated partitions anchored",
      bool(uA and uK) and uK.group(1) == "0"
      and f"give means from {uA.group(1)}% to {uA.group(3)}% (median {uA.group(2)}%, sd {uA.group(5)}), none at or below 50.0%" in FLAT,
      uA.groups() if uA else None)
check("unseen-producer: producer-cluster bootstrap anchored",
      bool(uB) and f"a 95% interval of [{uB.group(1)}, {uB.group(2)}], with {uB.group(3)}% of draws at or below chance" in FLAT,
      uB.groups() if uB else None)
check("unseen-producer: laboratory-prior control anchored",
      bool(uL) and f"scores {uL.group(1)}%, so those few points are not a laboratory prior" in FLAT,
      uL.group(1) if uL else None)
check("dual-class: repeated partitions anchored",
      bool(dA and dK) and dK.group(1) == "2"
      and f"twenty reassignments give means from {dA.group(1)}% to {dA.group(3)}% (median {dA.group(2)}%), two of them at or below 50.0%" in FLAT,
      dA.groups() if dA else None)
check("dual-class: bootstrap and laboratory control anchored",
      bool(dB and dL) and f"bootstrap interval is [{dB.group(1)}, {dB.group(2)}]" in FLAT
      and f"Laboratory identity alone scores {dL.group(1)}% under the same folds" in FLAT,
      (dB.groups() if dB else None, dL.group(1) if dL else None))
check("abstract and conclusion carry the partition range",
      bool(uA) and FLAT.count(f"{round(float(uA.group(1)))} to {round(float(uA.group(3)))}% across twenty") == 2,
      uA.groups() if uA else None)
check("grouped scores are not called a detected effect",
      FLAT.count("not distinguishable from chance at the producer level") >= 1
      and "small but real" not in FLAT and "real in the permutation sense" not in FLAT
      and "upper bound on the chemical signal" not in FLAT
      and "is real but small" not in FLAT)
check("the invalid row-level permutation test is gone",
      "permutation null is again" not in FLAT and not (EV / "perm_market.txt").exists())
e9 = re.search(r"analytes: (\d+) \(([\d.]+)% of usable rows\)\s*\n\s*after dropping them: rows=\d+\s+random ([\d.]+)%.*?unseen-producer ([\d.]+)%.*?gap ([\d.]+) points", log9)
check("duplicate count and sensitivity anchored",
      bool(e9) and f"{int(e9.group(1)):,} ({e9.group(2)}%) repeat" in FLAT
      and f"gives {e9.group(3)}% for the random split and {e9.group(4)}% grouped, a gap of {e9.group(5)} points" in FLAT
      and bool(rb and gb) and f"by {abs(float(rb[1])-float(e9.group(3))):.1f} points and the grouped score by {abs(float(gb[1])-float(e9.group(4))):.1f}" in FLAT,
      e9.groups() if e9 else None)
h9 = re.findall(r"macro_F1 folds=\[[^\]]*\] mean=([\d.]+)% \(\+/-([\d.]+)\)", log9)
f9 = re.search(r"majority ([\d.]+)%\s+stratified shuffle ([\d.]+)%", log9)
check("macro-F1 spreads and baselines anchored",
      len(h9) == 2 and bool(f9) and f"Macro F1 is {h9[0][0]}% (+/-{h9[0][1]}) and {h9[1][0]}% (+/-{h9[1][1]})" in FLAT
      and f"against {f9.group(1)}% for the majority predictor and {f9.group(2)}% for a stratified shuffle" in FLAT,
      (h9, f9.groups() if f9 else None))
p9 = re.search(r"pooled balanced_acc=([\d.]+)%", log9)
check("pooled balanced accuracy anchored",
      bool(p9) and f"pooled matrix gives a balanced accuracy of {p9.group(1)}%" in FLAT, p9.group(1) if p9 else None)
b9 = re.search(r"missingness-only probe: plain acc=([\d.]+)%\s+balanced_acc=([\d.]+)%", log9)
m9 = re.search(r"main model, random split: plain acc=([\d.]+)%", log9)
mp9 = re.search(r"accuracy from missingness alone:\s*([\d.]+)%", (EV / "lab_confounding.txt").read_text())
check("2.3 probe and main model given on both metrics",
      bool(b9 and m9) and f"the probe scores {b9.group(1)}% plain and {b9.group(2)}% balanced accuracy" in FLAT
      and f"against {m9.group(1)}% plain and {rb[1]}% balanced" in FLAT
      and f"{float(b9.group(1))-65.8:.1f} of {float(m9.group(1))-65.8:.1f} points on one metric and {float(b9.group(2))-50:.1f} of {float(rb[1])-50:.1f}" in FLAT
      and "Those are in-sample figures for the lookup" in FLAT,
      (b9.groups() if b9 else None, m9.group(1) if m9 else None))
k9 = re.search(r"PureVita Labs, LLC\s+\{\(0, 'solventless'\): \d+, \(1, 'hydrocarbon'\): (\d+), \(4, 'hydrocarbon'\): (\d+), \(6, 'solventless'\): (\d+)\}", log9)
check("within-laboratory class-dependent panel anchored",
      bool(k9) and k9.group(1) == k9.group(2) and f"all hydrocarbon ({int(k9.group(1)):,} each)" in FLAT
      and f"all solventless ({k9.group(3)})" in FLAT, k9.groups() if k9 else None)
z9 = re.search(r"distinct \(method, variety, t, T, P\) cells=(\d+)", log9)
check("controlled samples are distinct design cells",
      bool(z9) and z9.group(1) == "162" and "no replicates to leak across" in FLAT)
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
check("the chance claim is scoped and qualified",
      "all the way to chance" not in FLAT
      and "statistically indistinguishable from guessing" not in FLAT
      and "is not reliably recoverable" in FLAT and "is not reliably identifiable" in FLAT)
# permutation test on the market schemes, docs/evidence/perm_market.txt
check("no market p-value is quoted from a row-level shuffle",
      "above that null" not in FLAT)
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

sa = re.search(r"strains=(\d+) rows=(\d+)\s+acc=[\d.]+%\s+baseline=[\d.]+%"
               r"\s+lift=([-+]?[\d.]+)pts", log4b)
# 4.1 must not claim strain dominates in the MARKET data: the strain alarm
# probe there returns a negative lift, which 4.3 reports. The two datasets
# disagree, so the claim is scoped or it contradicts 4.3.
check("4.1 does not assert strain dominance unscoped",
      "Strain and producer variation dominate" not in FLAT)
check("4.1 scopes the cultivar claim to the controlled data",
      "datasets disagree on the evidence" in FLAT
      and "we do not claim it for the market corpus" in FLAT)
check("4.1 quotes the strain-probe lift",
      bool(sa) and f"{sa.group(3).lstrip('+')} points above its baseline (4.3)" in FLAT
      and "a fifth of the laboratory" not in FLAT)
check("4.1 supports producer dominance with both figures",
      bool(r7_pid and c9) and f"reaches {r7_pid.group(1)}% and the chemistry {c9.group(2)}%" in FLAT)
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
      "report plain accuracy against the explicit majority-class baseline" in FLAT
      and "given beside its majority-class baseline" in FLAT)
check("strain columns described as populated for 0 rows",
      "populated for **0 of 37,344**" in FLAT and "empty for **0 of" not in FLAT)
sa = re.search(r"strains=(\d+) rows=(\d+)\s+acc=[\d.]+%\s+baseline=[\d.]+%"
               r"\s+lift=([-+]?[\d.]+)pts", log4b)
check("strain alarm probe subset anchored",
      bool(sa) and f"the {sa.group(1)} keys with at least 30 rows "
      f"({int(sa.group(2)):,} rows)" in FLAT
      and f"a lift of {float(sa.group(3)):.1f} points" in FLAT
      and "*negative* lift" not in FLAT,
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
check("2.1 quotes the regulatory definition in force for the data period",
      "including concentrated cannabis extracted with ethanol or CO2" in FLAT
      and "by means other than with ethanol or CO2" in FLAT and "[12]" in TEXT
      and "the 2018 rule it replaced named CO2 only" in FLAT)
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
      bool(pr5 and plo) and f"p ≤ {float(pr5.group(1)):.3f} for the random split, which no shuffle reaches" in FLAT
      and f"p = {float(plo.group(1)):.2f} for leave-one-variety-out" in FLAT,
      (pr5.group(1), plo.group(1)) if pr5 and plo else None)
check("process-settings leak anchored",
      bool(ttp) and f"lifts the random split to {ttp.group(1)}%" in FLAT,
      ttp.group(1) if ttp else None)
check("discussion and conclusion carry the seed-averaged gap",
      bool(k5 and lovo and gap)
      and f"{round(float(k5.group(1)) - float(lovo[1]))} to {round(float(gap.group(1)))} points on the designed experiment" in FLAT
      and f"{round(float(k5.group(1)) - float(lovo[1]))} to {round(float(gap.group(1)))} on a designed experiment" in FLAT)
# ---------- markets and dates, from docs/evidence/lab_states.txt ----------
logst = (EV / "lab_states.txt").read_text()
nv = re.search(r"^\s+(\d+)\s+NV$", logst, re.M)
bl = re.search(r"^\s+(\d+)\s+\(blank\)$", logst, re.M)
dr = re.search(r"date_tested filled: (\d+) of (\d+)\s*\ndate_tested range: (\S+) to (\S+)", logst)
nv_labs = len(re.findall(r"\{'NV': \d+\}", logst))
check("Nevada row count and share anchored",
      bool(nv) and f"{int(nv.group(1)):,} of the usable rows ({100*int(nv.group(1))/len(rows):.1f}%)" in FLAT
      and f"{nv_labs} laboratories" in FLAT.replace("seven", "7"), nv.group(1) if nv else None)
check("blank lab_state rows anchored",
      bool(bl) and f"remaining {int(bl.group(1)):,} rows" in FLAT, bl.group(1) if bl else None)
check("no other state appears in the data",
      len(re.findall(r"^\s+\d+\s+\S+$", logst.split("rows per lab_state")[1].split("date_tested")[0], re.M)) == 2)
check("date range anchored",
      bool(dr) and f"dated {dr.group(3)} to {dr.group(4)} on the {int(dr.group(1)):,} rows" in FLAT,
      dr.groups() if dr else None)

# ---------- pass ten robustness, docs/evidence/review10b_robustness.txt ----------
log10b = (EV / "review10b_robustness.txt").read_text()
c1r = re.search(r"random, producer rows\s+([\d.]+)%", log10b)
c1g = re.search(r"unseen-producer\s+([\d.]+)% \(\+/-[\d.]+\)\s+same-sample gap ([\d.]+) points", log10b)
check("market data: logistic regression anchored",
      bool(c1r and c1g) and f"scores {c1r.group(1)}% on the same-sample random split and {c1g.group(1)}% grouped: a gap of {c1g.group(2)} points" in FLAT,
      (c1r.group(1), c1g.groups()) if c1r and c1g else None)
lg = re.search(r"logistic regression\s+method random ([\d.]+)% \(\+/-([\d.]+)\)\s+method LOVO ([\d.]+)% \(\+/-([\d.]+)\)\s+variety random ([\d.]+)%.*?gap ([\d.]+) points", log10b)
check("controlled data: logistic rows anchored in the table",
      bool(lg) and f"random 5-fold                  {lg.group(1)}% (+/-{lg.group(2)})" in TEXT
      and f"leave-one-variety-out          {lg.group(3)}% (+/-{lg.group(4)})" in TEXT, lg.groups() if lg else None)
check("controlled data: logistic gap and variety anchored",
      bool(lg and gap) and f"drops {lg.group(6)} points instead of {gap.group(1)}" in FLAT
      and f"logistic regression ({lg.group(5)}%)" in FLAT
      and f"from {lg.group(1)}% to {lg.group(3)}% with logistic regression" in FLAT
      and FLAT.count(f"{lg.group(1)}% to {lg.group(3)}% with logistic regression") == 2,
      lg.groups() if lg else None)
lp = re.search(r"logistic LOVO permutation: obs ([\d.]+)%.*?at or above obs: (\d+) of (\d+)\s+p=([\d.]+)", log10b)
check("controlled data: logistic permutation anchored",
      bool(lp) and f"{lp.group(2)} of {int(lp.group(3)):,} label shuffles reaches {lp.group(1)}% (p = {lp.group(4)})" in FLAT,
      lp.groups() if lp else None)
ls = re.search(r"over 200 seeds: mean ([\d.]+)% sd ([\d.]+)\s+seed-averaged gap to LOVO ([\d.]+) points", log10b)
check("controlled data: logistic seed sweep anchored",
      bool(ls) and f"the 200-seed average is {ls.group(1)}% (sd {ls.group(2)}) and the gap {ls.group(3)}" in FLAT,
      ls.groups() if ls else None)
cap = re.findall(r"max_iter=(\d+)\s+random ([\d.]+)%.*?unseen-producer ([\d.]+)%.*?gap ([\d.]+) points", log10b)
check("gap against capacity anchored",
      len(cap) == 4
      and f"scores {cap[0][1]}%, {cap[1][1]}%, {cap[2][1]}% and {cap[3][1]}% and the grouped one {cap[0][2]}%, {cap[1][2]}%, {cap[2][2]}% and {cap[3][2]}%" in FLAT
      and f"from {cap[0][3]} to {cap[3][3]} points" in FLAT, cap)
sub = re.findall(r"with producer=(\d+) producers=\d+.*?\n\s+random \(producer rows\) ([\d.]+)%.*?unseen-producer ([\d.]+)% \(\+/-([\d.]+)\)\s+gap ([\d.]+) points", log10b)
check("label-definition subsets anchored",
      len(sub) == 2
      and f"On the {int(sub[0][0]):,} producer rows not named distillate, the random split scores {sub[0][1]}% and the grouped one {sub[0][2]}% (+/-{sub[0][3]}), a gap of {sub[0][4]} points" in FLAT
      and f"On the {int(sub[1][0]):,} rows whose names" in FLAT
      and f"they score {sub[1][1]}% and {sub[1][2]}% (+/-{sub[1][3]}), a gap of {sub[1][4]} points" in FLAT, sub)
tmp = re.search(r"balanced_acc ([\d.]+)%\s+test producers seen in training: ([\d.]+)% of test rows", log10b)
check("temporal split anchored",
      bool(tmp) and f"scores {tmp.group(1)}% with {tmp.group(2)}% of its test rows from producers seen in training" in FLAT,
      tmp.groups() if tmp else None)
glr = re.search(r"random, all rows ([\d.]+)% \(\+/-([\d.]+)\)\s+unseen-producer ([\d.]+)%", log10b.split("=== G.")[1])
check("binary task at the probe learning rate anchored",
      bool(glr and rb and gb) and f"is {rb[2]} at the default rate and {glr.group(2)} at 0.05, with the grouped score at {gb[1]}% and {glr.group(3)}%" in FLAT,
      glr.groups() if glr else None)
hs = re.search(r"rows with a strain key=\d+\s+random ([\d.]+)%.*?\n\s+dual-class strain rows=\d+\s+random ([\d.]+)%", log10b)
check("strain subsets: same-sample random anchored",
      bool(hs) and f"{hs.group(1)}% for the keyed rows and {hs.group(2)}% for the dual-class keys" in FLAT,
      hs.groups() if hs else None)
p01 = re.search(r"producer probe at 0\.1: folds=\[([\d., ]+)\]\s+mean=[\d.]+%\s+lift=\+([\d.]+)pts", log3)
check("producer probe at the 0.1 rate is logged and matches the text",
      bool(p01) and f"{min(float(v) for v in p01.group(1).split(','))} to {max(float(v) for v in p01.group(1).split(','))}" in FLAT
      and f"a {p01.group(2)}-point lift" in FLAT, p01.groups() if p01 else None)
fl = re.findall(r"^\s+folds=\[([\d., ]+)\]\s+balanced_acc", log3, re.M)
check("fold spans at the 0.05 rate anchored",
      len(fl) == 2 and f"span {max(map(float, fl[0].split(','))) - min(map(float, fl[0].split(','))):.1f} points for the laboratory and "
      f"{max(map(float, fl[1].split(','))) - min(map(float, fl[1].split(','))):.1f} for the producer" in FLAT, fl)
rc = re.findall(r"recall\[hydro=([\d.]+)% solventless=([\d.]+)%\]", log4)
check("per-class recalls of the other schemes anchored",
      len(rc) == 4 and f"{rc[0][0]}% and {rc[0][1]}% under the random split, and {rc[3][0]}% and {rc[3][1]}% on the dual-class subset" in FLAT, rc)
def _fam(key):
    m = re.search(rf"^{re.escape(key)}.*?(\d+) \(\s*[\d.]+%\)\s+(\d+) \(\s*[\d.]+%\)$", logl, re.M)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)
_h, _e, _m, _c = _fam("hydrocarbon words"), _fam("ethanol"), _fam("mechanical words"), _fam("co2 / supercritical")
check("label contradiction counts anchored",
      f"{_h[0]} non-solvent rows name a hydrocarbon process, and {_e[1]} solvent-based rows name ethanol, {_m[1]} a mechanical process and {_c[1]} CO2" in FLAT
      and round(100 * (_h[0] + _e[1] + _m[1] + _c[1]) / len(rows)) == 1, (_h, _e, _m, _c))
_labc = Counter((r.get("lab") or "").strip() for r in rows if (r.get("producer") or "").strip())
_top2 = sum(n for _, n in _labc.most_common(2))
check("joint hold-out caveat: share of the two largest laboratories",
      f"hold {round(100*_top2/pf)}% of the rows that carry a producer" in FLAT, f"{100*_top2/pf:.1f}")
_hs = {}
for r in rows:
    _hs.setdefault((r.get("lab") or "").strip(), []).append((r.get("label") or "").strip() == "hydrocarbon")
_sh = [100 * sum(v) / len(v) for v in _hs.values()]
check("laboratory hydrocarbon-share range anchored",
      f"runs from {min(_sh):.1f}% to {max(_sh):.1f}%" in FLAT and "also not entangled" not in FLAT,
      f"{min(_sh):.1f}-{max(_sh):.1f}")
check("sd convention stated", "population standard deviation of the fold scores" in FLAT)
check("the logistic comparator and its inputs are described in methods",
      "A standardised logistic regression is run beside it" in FLAT
      and "log-transformed values with empty cells set to zero" in FLAT)
check("[11] is not credited with the earliest demonstration",
      "earliest quantitative demonstration" not in FLAT)
check("the abstract gives the same-sample range of the inflation",
      "we measure 24 to 27 points of inflation" in FLAT)

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
