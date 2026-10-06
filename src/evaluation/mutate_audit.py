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

REPO = Path("/home/arthur/cannabis-extraction-ds")
SANDBOX = Path("/tmp/mutation-sandbox")
PY = REPO / ".venv/bin/python"

ORIGINAL = (REPO / "PREPRINT.md").read_text()


def build_sandbox():
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    (SANDBOX / "src/evaluation").mkdir(parents=True)
    for d in ("data", "docs"):
        (SANDBOX / d).symlink_to(REPO / d)
    shutil.copy(REPO / "src/evaluation/audit_preprint.py",
                SANDBOX / "src/evaluation/audit_preprint.py")


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
    ("random accuracy 81.0 -> 81.4", "81.0% balanced accuracy",
     "81.4% balanced accuracy"),
    ("grouped accuracy 54.0 -> 55.0", "unseen-producer              54.0%",
     "unseen-producer              55.0%"),
    ("dual-class 54.6 -> 55.6", "dual-class + unseen          54.6%",
     "dual-class + unseen          55.6%"),
    ("lab lift 15.3 -> 16.3", "+15.3 pts", "+16.3 pts"),
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
    ("same-sample random 78.5 -> 79.5", "rows* scores 78.5%",
     "rows* scores 79.5%"),
    ("protocol gap 24.4 -> 25.4", "**24.4 points**", "**25.4 points**"),
    ("CI lower 46.3 -> 47.3", "[46.3, 61.8]", "[47.3, 61.8]"),
    ("producer-ID-only 84.4 -> 85.4", "reaches 84.4% balanced",
     "reaches 85.4% balanced"),
    # coverage table
    ("limonene detected 40.7 -> 41.7", "75.7%   81.1%           40.7%   65.6%",
     "75.7%   81.1%           41.7%   65.6%"),
    # label agreement
    ("agreement 99.3 -> 99.4", "4,254 of 4,284 rows, 99.3%",
     "4,254 of 4,284 rows, 99.4%"),
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
     "rows they cover:                  4123 (11.03%)",
     "rows they cover:                  4124 (11.03%)"),
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
    # prose hygiene: an em-dash must be caught
    ("em-dash introduced", "The task is a good\ntest case for three reasons.",
     "The task is a good test case \u2014 for three reasons."),
]

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
    if old not in ORIGINAL:
        print(f"    {'ANCHOR MISSING':<16} {label}")
        survived.append((label, "anchor not found in manuscript"))
        continue
    mutated = ORIGINAL.replace(old, new, 1)
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
