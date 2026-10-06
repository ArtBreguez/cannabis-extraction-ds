"""Mutation testing for audit_preprint.py.

57 checks passing proves nothing by itself. A check can be vacuous: `"54.0%" in
TEXT` passes whether or not 54.0 is the right number, and a check whose
reference value was copied from the prose cannot detect that the prose is
wrong. That is exactly the bug pass seven found, where the audit and the
manuscript agreed on 55 dual-class producers while the committed log said 53.

So: plant a wrong number in the manuscript, run the audit, and require it to
FAIL. Any mutation that survives is a hole in the audit, and a number a referee
could be the first to catch.

Isolation: the audit resolves ROOT from its own file location, so a sandbox is
built as
    /tmp/mutation-sandbox/
        PREPRINT.md            <- mutated copy
        data/ docs/            <- symlinks to the real ones
        src/evaluation/audit_preprint.py  <- copy
which leaves the real repository untouched.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SANDBOX = Path("/tmp/mutation-sandbox")
PY = REPO / ".venv/bin/python"

ORIGINAL = (REPO / "PREPRINT.md").read_text()


def build_sandbox():
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    (SANDBOX / "src/evaluation").mkdir(parents=True)
    # reports/ holds Figure 1, which the audit compares against the logs
    for d in ("data", "docs", "reports"):
        (SANDBOX / d).symlink_to(REPO / d)
    # the audit imports make_figure.parse(), so the figure script travels too
    for f in ("audit_preprint.py", "make_figure.py", "__init__.py"):
        shutil.copy(REPO / "src/evaluation" / f, SANDBOX / "src/evaluation" / f)
    shutil.copy(REPO / "src/__init__.py", SANDBOX / "src/__init__.py")


def run_audit(text):
    (SANDBOX / "PREPRINT.md").write_text(text)
    r = subprocess.run(
        [str(PY), str(SANDBOX / "src/evaluation/audit_preprint.py")],
        capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    m = re.search(r"(\d+)/(\d+) checks passed", out)
    if not m:
        return None, out
    passed, total = int(m.group(1)), int(m.group(2))
    return (passed, total), out


# Each mutation is (label, old, new). They are deliberately SMALL: a referee
# catches a wild number anyway, the dangerous error is the plausible one.
MUTATIONS = [
    # headline results
    ("grouped accuracy 54.0 -> 55.0", "unseen-producer              54.0%",
     "unseen-producer              55.0%"),
    ("dual-class 54.6 -> 55.6", "dual-class + unseen          54.6%",
     "dual-class + unseen          55.6%"),
    ("lab lift 52.0 -> 53.0", "+52.0 pts", "+53.0 pts"),
    ("missingness-only lab probe 57.3 -> 58.3", "reaches 57.3% on the same folds",
     "reaches 58.3% on the same folds"),
    ("producer-grouped lab probe 85.4 -> 86.4", "85.4% (+/-3.3) against",
     "86.4% (+/-3.3) against"),
    ("strain lift 9.9 -> 10.9 in 4.3", "a lift of 9.9 points", "a lift of 10.9 points"),
    ("solventless recall 32.3 -> 33.3", "recall is **32.3%**",
     "recall is **33.3%**"),
    # population figures, the class of bug pass seven found
    ("usable rows 37,344 -> 37,345", "leaving **37,344 rows**",
     "leaving **37,345 rows**"),
    ("hydrocarbon 24,573 -> 24,574", "hydrocarbon   24,573",
     "hydrocarbon   24,574"),
    ("solventless 12,771 -> 12,772", "solventless   12,771",
     "solventless   12,772"),
    ("producer rows 33,229 -> 33,230", "producer      33,229 rows",
     "producer      33,230 rows"),
    ("dual producers 53 -> 54", "the 53 producers that make",
     "the 54 producers that make"),
    ("dual rows 27,751 -> 27,752", "covers 27,751 rows",
     "covers 27,752 rows"),
    ("dual share 74.3 -> 75.3", "74.3% of the usable data",
     "75.3% of the usable data"),
    ("dual split 17,312 -> 17,313", "17,312 / 10,439", "17,313 / 10,439"),
    # the reduction table
    ("declared 37,310 -> 37,311", "extraction family    37,310",
     "extraction family    37,311"),
    ("undeclared 102,404 -> 102,405", "does not                         102,404",
     "does not                         102,405"),
    ("pre-roll share 0.17 -> 0.18", "0.17% of the labelled data",
     "0.18% of the labelled data"),
    # statistical qualifications added in pass seven
    ("protocol gap 24.4 -> 25.4", "**24.4 points**", "**25.4 points**"),
    ("CI lower 46.3 -> 47.3", "[46.3, 61.8]", "[47.3, 61.8]"),
    ("producer-ID-only 84.4 -> 85.4", "reaches 84.4% balanced",
     "reaches 85.4% balanced"),
    # coverage table
    ("limonene detected 40.7 -> 41.7", "75.7%   81.1%           40.7%   65.6%",
     "75.7%   81.1%           41.7%   65.6%"),
    # label agreement
    # 2.3 pre-check: the defect was a probe narrower than the feature set,
    # which reported a +0.0 lift. Each figure must be pinned to the log.
    ("missingness accuracy 69.9 -> 70.9", "reaches 69.9% accuracy",
     "reaches 70.9% accuracy"),
    ("missingness lift 4.1 -> 3.1", "lift of **+4.1 points**",
     "lift of **+3.1 points**"),
    ("lift silently reverted to +0.0", "lift of **+4.1 points**",
     "lift of **+0.0 points**"),
    ("pure strata 7 of 13 -> 6 of 13",
     "patterns that are 100% one class: 7 of 13",
     "patterns that are 100% one class: 6 of 13"),
    ("pure stratum rows 4123 -> 4124",
     "rows they cover:                  4123 (11.04%)",
     "rows they cover:                  4124 (11.04%)"),
    ("pure stratum share reverted to the 37,374 denominator",
     "rows they cover:                  4123 (11.04%)",
     "rows they cover:                  4123 (11.03%)"),
    ("largest lab count reverted to the 37,374 population",
     "15,018 and 12,531 rows", "15,035 and 12,539 rows"),
    # pass eight: figures that previously had no committed log
    ("ablation cost 0.3 -> 0.2", "costs 0.3 points of balanced accuracy",
     "costs 0.2 points of balanced accuracy"),
    ("dual-class CI lower 51.0 -> 50.0", "[51.0, 58.3] that excludes 50.0%",
     "[50.0, 58.3] that excludes 50.0%"),
    ("dual-class p 0.024 -> 0.24", "excludes 50.0% (p = 0.024)", "excludes 50.0% (p = 0.24)"),
    ("dual-class fold 50.9 -> 51.9", "56.4 and 50.9", "56.4 and 51.9"),

    ("dual-class same-sample random 80.3 -> 81.3", "against 80.3% (+/-0.7)", "against 81.3% (+/-0.7)"),

    ("pooled balanced accuracy 51.6 -> 52.6", "balanced accuracy of 51.6%", "balanced accuracy of 52.6%"),
    ("abstract partition range 51 -> 53", "51 to 57% across twenty fold assignments",
     "53 to 57% across twenty fold assignments"),
    ("logistic LOVO 46.3 -> 47.3 in the table", "leave-one-variety-out          46.3% (+/-10.2)",
     "leave-one-variety-out          47.3% (+/-10.2)"),
    ("logistic permutation p 0.002 -> 0.02", "reaches 46.3% (p = 0.002)", "reaches 46.3% (p = 0.02)"),
    ("regulation reverted to CO2 only", "including concentrated cannabis extracted with ethanol or CO2\" against",
     "including concentrated cannabis extracted with CO2\" against"),
    ("censoring direction flipped back", "heavily left-censored",
     "heavily right-censored"),
    ("strain columns 'empty for 0' wording back",
     "populated for **0 of 37,344**", "empty for **0 of 37,344**"),
    ("alarm max_iter 120 -> 200", "at `max_iter=120`", "at `max_iter=200`"),
    ("strain probe subset 109 -> 108", "the 109 keys with at least 30 rows",
     "the 108 keys with at least 30 rows"),
    ("reduction table total 37,374 -> 37,375",
     "total labelled rows                              37,374",
     "total labelled rows                              37,375"),
    ("regex variant count 1,445 -> 1,455", "`non-solvent concentrate` (1,445 rows)",
     "`non-solvent concentrate` (1,455 rows)"),
    ("name-path scope 11.5 -> 12.5", "11.5% of labelled rows",
     "12.5% of labelled rows"),
    ("intro reverts to rosin-vs-BHO framing",
     "separating concentrates declared *non-solvent based*",
     "separating solventless extracts (rosin, live rosin, hash rosin) from hydrocarbon extracts, declared *non-solvent based*"),
    ("seed-sweep mean 54.3 -> 55.3", "average 54.3% (sd 2.7)",
     "average 55.3% (sd 2.7)"),
    ("seed-robust gap 15.4 -> 16.4", "is 15.4 points rather than",
     "is 16.4 points rather than"),
    ("random-group control 54.0 -> 55.0", "averages 54.0% over 200 draws",
     "averages 55.0% over 200 draws"),
    ("LOVO permutation p 0.11 -> 0.01", "p = 0.11 for leave-one-variety-out",
     "p = 0.01 for leave-one-variety-out"),
    ("t/T/P leak 88.3 -> 98.3", "split to 88.3%", "split to 98.3%"),
    ("conclusion drops the seed-averaged gap", "15 to 18 on a designed experiment",
     "18 on a designed experiment"),
    # ---- pass eleven: anchors moved by the rewrite, and the new figures ----
    ("P11 table 9 random row 81.0 -> 81.4", "random (optimistic)          81.0%",
     "random (optimistic)          81.4%"),
    ("P11 lab probe 92.3 -> 93.3 in the abstract", "laboratory at 92.3% against 40.3%",
     "laboratory at 93.3% against 40.3%"),
    ("P11 same-sample random 78.5 -> 79.5", "the random split scores 78.5%, and that pair",
     "the random split scores 79.5%, and that pair"),
    ("P11 macro-F1 majority baseline 39.8 -> 40.8", "against 39.8% and 38.4% for the majority predictor",
     "against 40.8% and 38.4% for the majority predictor"),
    ("P11 PureVita pairs 1,230 -> 1,231", "(1,230 of them at `PureVita`)", "(1,231 of them at `PureVita`)"),
    ("P11 partition range 51.3 -> 52.3", "means from 51.3% to 56.6%", "means from 52.3% to 56.6%"),
    ("P11 pooled bootstrap lower bound 47.3 -> 50.3", "[47.3, 56.6] for the pooled score",
     "[50.3, 56.6] for the pooled score"),
    ("P11 fold-mean bootstrap 50.2 -> 49.2", "[50.2, 58.9] for the fold mean", "[49.2, 58.9] for the fold mean"),
    ("P11 dual-class bootstrap 45.8 -> 50.8", "spans [45.8, 59.8] for the pooled balanced accuracy",
     "spans [50.8, 59.8] for the pooled balanced accuracy"),
    ("P11 'indistinguishable from chance' reinstated",
     "The claim is about the size of the inflation, not that nothing transfers.",
     "The held-out score is not distinguishable from chance at the producer level."),
    ("P11 logistic market gap 11.0 -> 12.0", "11.0 for an unweighted logistic regression",
     "12.0 for an unweighted logistic regression"),
    ("P11 capacity gap 26.1 -> 27.1", "from 21.1 to 26.1", "from 21.1 to 27.1"),
    ("P11 named-rows grouped 60.3 -> 61.3", "77.0%    60.3%     16.8", "77.0%    61.3%     16.8"),
    ("P11 the false 'no gate passed' sentence reinstated",
     "so on that thresholded metric neither beats a stratified guess.",
     "No gate that requires beating chance by more than the between-fold spread is passed."),
    ("P11 reseed gap 24.5 -> 24.4", "24.6, 24.5, 24.6", "24.6, 24.4, 24.6"),
    ("P11 composition: distillate share 51.5 -> 50.5", "5,805 (51.5%)", "5,805 (50.5%)"),
    ("P11 composition: mechanical count 559 -> 569", "559 ( 5.0%)", "569 ( 5.0%)"),
    ("P11 held-out AUC 0.625 -> 0.725 in table 10", "54.0% (+/-5.6)  0.625", "54.0% (+/-5.6)  0.725"),
    ("P11 pooled AUC interval lower 0.535 -> 0.435", "0.610 [0.535, 0.698]", "0.610 [0.435, 0.698]"),
    ("P11 surviving share 0.125 -> 0.225", "0.125 of 0.394", "0.225 of 0.394"),
    ("P11 weighted logistic 61.1 -> 62.1", "whose held-out 61.1% is the highest", "whose held-out 62.1% is the highest"),
    ("P11 merged pairs random 80.4 -> 81.4", "with those pairs merged it is 80.4%",
     "with those pairs merged it is 81.4%"),
    ("P11 split-record count 1,303 -> 1,304", "1,303 samples from the producer-less laboratories",
     "1,304 samples from the producer-less laboratories"),
    ("P11 ethanol table 133 -> 143", "2023          13           133", "2023          13           143"),
    ("P11 positive control 76.4 -> 77.4", "is recovered at 76.4%", "is recovered at 77.4%"),
    ("P11 permutation floor 0.0033 -> 0.003", "p = 0.0033 for the random split", "p = 0.003 for the random split"),
    ("P11 class-weighted partitions 53.0 -> 54.0", "the class-weighted one 53.0% to 59.9%",
     "the class-weighted one 54.0% to 59.9%"),
    ("P11 dual-class AUC 0.632 -> 0.732", "its AUC falls from 0.899 to 0.632", "its AUC falls from 0.899 to 0.732"),
    # ---- pass twelve ----
    ("P12 agreement 99.3 -> 99.4", "4,259 of 4,289 rows, 99.3%", "4,259 of 4,289 rows, 99.4%"),
    ("P12 agreement count 4,259 -> 4,254", "agree on 4,259 of 4,289", "agree on 4,254 of 4,289"),
    ("P12 pre-roll random 80.6 -> 80.7", "without them the all-rows random split is 80.6%",
     "without them the all-rows random split is 80.7%"),
    ("P12 duplicate count 1,813 -> 1,814", "1,813 usable rows (4.9%) repeat", "1,814 usable rows (4.9%) repeat"),
    ("P12 duplicates naming a producer 419 -> 429", "only 419 of them name a producer",
     "only 429 of them name a producer"),
    ("P12 headline random 78.5 -> 79.5 in the abstract", "scores 78.5% balanced accuracy and a ROC AUC of",
     "scores 79.5% balanced accuracy and a ROC AUC of"),
    ("P12 dedup held-out 55.2 -> 56.2", "78.9%    55.2%     23.7", "78.9%    56.2%     23.7"),
    ("P12 temporal split 58.3 -> 59.3", "scores 58.3% with an AUC of 0.754", "scores 59.3% with an AUC of 0.754"),
    ("P12 temporal AUC 0.754 -> 0.854", "0.754 against 0.625", "0.854 against 0.625"),
    ("P12 the old temporal claim reinstated", "So the two hold-outs are not interchangeable",
     "So time alone costs far less than holding out the producer"),
    ("P12 abstract overstatement 24 -> 27", "overstates it by 8 to 24 points", "overstates it by 8 to 27 points"),
    ("P12 abstract held-out AUC 0.62 -> 0.72", "leaves 54.0% and 0.62.", "leaves 54.0% and 0.72."),
    ("P12 abstract pooled AUC 0.61 -> 0.71", "the pooled held-out AUC is 0.61,", "the pooled held-out AUC is 0.71,"),
    ("P12 conclusion AUC interval 0.54 -> 0.44", "interval of 0.54 to 0.70 that excludes chance",
     "interval of 0.44 to 0.70 that excludes chance"),
    ("P12 'less than a third' -> 'more than a third'", "keeps less than a third of its above-chance AUC, and a logistic",
     "keeps more than a third of its above-chance AUC, and a logistic"),
    ("P12 logistic share 'about half' -> 'about a third'", "0.134 of 0.253", "0.084 of 0.253"),
    ("P12 terpene share 72.0 -> 82.0", "72.0% in 2023", "82.0% in 2023"),
    ("P12 largest producer 13.2 -> 14.2", "the largest producer holds 13.2%", "the largest producer holds 14.2%"),
    ("P12 title reverted to 'collapses'", "sharply reduces the apparent accuracy of", "collapses"),
    ("P12 title back to certificates", "in public cannabis laboratory data", "in public cannabis certificates of analysis"),
    ("P12 stable-period gap 28.4 -> 18.4", "28.4 points on the rows tested in 2020 to 2022",
     "18.4 points on the rows tested in 2020 to 2022"),
    ("P12 terpene-only rows 79.4 -> 80.4", "79.4%    55.1%     24.3", "80.4%    55.1%     24.3"),
    ("P12 producer spellings 260 -> 360 rows", "spellings of one company (260 rows)",
     "spellings of one company (360 rows)"),
    ("P12 data source back to regulatory portals",
     "public records requests and from certificates of analysis published online",
     "state regulatory portals and published COAs"),
    ("P12 random-split recalls back to the all-rows run", "89.8% and 67.1% under the random split on the same rows",
     "90.7% and 71.3% under the random split on the same rows"),
    ("P12 dual class-weighted interval 0.509 -> 0.409", "[0.509, 0.723]", "[0.409, 0.723]"),
    ("P12 table 9 producer-rows random 78.5 -> 79.5", "random, producer rows        78.5%",
     "random, producer rows        79.5%"),
    ("P12 partitions AUC 0.603 -> 0.703", "AUC 0.603 to 0.658", "AUC 0.703 to 0.658"),
    ("P12 method ratio under the random split 1.7 -> 1.2", "against 1.7 and 1.6 times chance for the method",
     "against 1.2 and 1.4 times chance for the method"),
    ("P12 terpene panel 2024 claim reinstated", "to 85.8% in 2022 and 72.0% in 2023.",
     "to 84.9% in 2022, 47.7% in 2023 and none in 2024."),
    ("reference [1] journal reverted",
     "*European Archives of Psychiatry and\nClinical Neuroscience*",
     "*Journal of\nCannabis Research*"),
    # 2.2 disclosures
    ("total_terpenes R2 0.984 -> 0.994", "R² = 0.984", "R² = 0.994"),
    ("total_terpenes mask 33,229 -> 33,228", "33,229 / 33,229 rows",
     "33,228 / 33,229 rows"),
    ("analyte columns 34 -> 35", "carries 34 analyte columns",
     "carries 35 analyte columns"),
    ("empty dropped Fifteen -> Sixteen", "Fifteen are empty",
     "Sixteen are empty"),
    # 3.2 recall reconciliation: the pooled figure must match the matrix
    ("pooled recall 29.6 -> 30.6", "3,332 / 11,262 = **29.6%**",
     "3,332 / 11,262 = **30.6%**"),
    # 4.3 strain: denominator repair and the two previously-withheld runs
    ("junk-key share 22.6 -> 23.6", "(22.6% of keys)", "(23.6% of keys)"),
    ("no-signal share 32.4 -> 33.4", "(32.4% of usable)",
     "(33.4% of usable)"),
    ("the bad 37.1% ratio reinstated", "nearly a quarter of the keys",
     "37.1% of the keys"),
    ("unseen-strain result withheld", "gives 66.2% balanced accuracy",
     "gives a figure we do not report"),
    ("dual-strain result withheld", "gives 56.9% (+/-11.4)",
     "gives a figure we do not report"),
    ("dual-strain subset size 9,598 -> 9,599", "9,598 rows", "9,599 rows"),
    # 4.1 scope: strain dominance must not be asserted for the market corpus
    ("4.1 strain dominance unscoped again",
     "Producer variation dominates\nwhatever method signal exists",
     "Strain and producer variation dominate\nwhatever method signal exists"),
    ("4.1 drops the strain-probe lift", "9.9 points above its baseline (4.3)",
     "a weak signal (4.3)"),
    ("4.1 producer-ID figure 84.4 -> 85.4", "reaches 84.4% and the chemistry",
     "reaches 85.4% and the chemistry"),
    # prose hygiene: an em-dash must be caught
    ("em-dash introduced", "The task is a good test case for three reasons.",
     "The task is a good test case \u2014 for three reasons."),
]

if len(sys.argv) > 1:
    MUTATIONS = [mu for mu in MUTATIONS if any(arg in mu[0] for arg in sys.argv[1:])]
    print(f"  filter {sys.argv[1:]}: {len(MUTATIONS)} mutations selected")

print("  building sandbox")
build_sandbox()

base, base_out = run_audit(ORIGINAL)
if base is None:
    print("  BASELINE DID NOT RUN:")
    print(base_out[-2000:])
    sys.exit(1)
print(f"  baseline: {base[0]}/{base[1]} checks passed")
if base[0] != base[1]:
    print("  baseline is not clean, fix that before mutating")
    sys.exit(1)

print(f"\n  running {len(MUTATIONS)} mutations\n")
survived = []
for label, old, new in MUTATIONS:
    # Anchors are matched across line breaks: the manuscript is reflowed at
    # 80 columns, so a phrase may wrap differently from one pass to the next.
    pat = r"\s+".join(re.escape(w) for w in old.split())
    hit = re.search(pat, ORIGINAL)
    if not hit:
        print(f"    {'ANCHOR MISSING':<16} {label}")
        survived.append((label, "anchor not found in manuscript"))
        continue
    mutated = ORIGINAL[:hit.start()] + new + ORIGINAL[hit.end():]
    res, out = run_audit(mutated)
    if res is None:
        print(f"    {'AUDIT CRASHED':<16} {label}")
        survived.append((label, "audit crashed"))
        continue
    passed, total = res
    if passed < total:
        caught = re.findall(r"FAIL\s+(.+?)\s+\(measured", out)
        print(f"    {'caught':<16} {label:<38} "
              f"({total-passed} check(s): {', '.join(caught[:2])})")
    else:
        print(f"    {'SURVIVED':<16} {label}")
        survived.append((label, "audit still passed"))

print(f"\n  === result ===")
print(f"    mutations killed  : {len(MUTATIONS) - len(survived)}"
      f"/{len(MUTATIONS)}")
if survived:
    print(f"    SURVIVORS, each a hole in the audit:")
    for label, why in survived:
        print(f"      - {label}: {why}")
else:
    print("    no survivors: every planted error was detected")

# restore, paranoia: the real file must be untouched
assert (REPO / "PREPRINT.md").read_text() == ORIGINAL, \
    "the real manuscript was modified, which must never happen"
print("\n  real manuscript verified unchanged")
