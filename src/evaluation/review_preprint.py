"""Adversarial read of the preprint: what would a reviewer attack?

The audit confirms arithmetic. It cannot see an overclaim, a missing control, or
a sentence that outruns its evidence. This encodes the specific objections a
referee in chemometrics or cannabis science would raise, and checks whether the
manuscript already answers each one.

Every item is a real attack surface, not a style note.
"""
import re
from pathlib import Path

ROOT = Path("/home/arthur/cannabis-extraction-ds")
T = (ROOT / "PREPRINT.md").read_text()

findings = []


def need(label, present, why):
    if not present:
        findings.append((label, why))


# 1. "Your model is just weak / undertuned."
need("no-hyperparameter-defence",
     "held constant across all schemes" in T or "not the estimator" in T,
     "A referee will say the collapse is a weak model, not leakage. The text "
     "must state the estimator is identical across schemes, so the DELTA "
     "cannot be a tuning artefact.")

# 2. Did we tune on the test fold ourselves?
need("own-leakage-disclosure",
     "fixed seed" in T and "early stopping disabled" in T,
     "If we tuned anything per fold we commit the sin we are reporting. State "
     "that no hyperparameter search was run.")

# 3. Class imbalance could explain the balanced-accuracy drop.
need("baselines-reported",
     "most-frequent" in T and "stratified" in T,
     "Both trivial baselines must be printed or the 54% is unanchored.")

# 4. 19 analytes is a thin panel; is the negative just low dimensionality?
need("feature-poverty-acknowledged",
     "richer assay" in T or "different experiment" in T,
     "Must concede the negative is about THESE features, not about chemistry.")

# 5. The dual-class subset could be unrepresentative.
need("dual-subset-sized",
     "77.4%" in T and "18,470" in T,
     "Size and class balance of the strict subset must be given, or a referee "
     "assumes it is a small unrepresentative corner.")

# 6. Are we accusing the cited papers of being wrong?
need("no-accusation",
     "may well hold" in T or "not a claim about the correctness" in T,
     "Attacking [1][2][3]'s conclusions is not supported by our data and will "
     "draw a hostile review. Separate 'protocol inflates' from 'result false'.")

# 7. Label quality.
need("label-provenance",
     "source-declared" in T and "99.3%" in T,
     "The whole study rests on the label. Provenance and agreement rate needed.")

# 8. Multiple-comparisons / fishing.
need("no-fishing",
     "before modelling" in T or "before the main task" in T,
     "The alarm models must be presented as pre-registered-in-spirit, run "
     "BEFORE the main task, or they look like post-hoc rescue.")

# 9. Does the controlled replication actually transfer?
need("transfer-caveat",
     "lab methods" in T.lower() or "laboratory extraction methods" in T,
     "Zenodo uses maceration/ultrasound/CO2, NOT rosin vs BHO. If the text "
     "implies it replicates the same task, that is an overclaim.")

# 10. Conflict of interest.
need("coi-declared",
     "Competing interests" in T and "HiddenTerps" in T,
     "Author sells solventless content. Undeclared COI on a solventless paper "
     "is a desk-reject risk.")

# 11. Reproducibility.
need("code-available",
     "rederive_paper_numbers" in T,
     "A methods-criticism paper that is not reproducible is self-defeating.")

# 12. The strain confounder is the obvious hole.
need("strain-handled",
     "0 of 37,374" in T and "data-availability limit" in T,
     "Strain is the first thing a reviewer asks. Must be addressed as a limit, "
     "not ignored, and the noisy-key result must not be leaned on.")

# 13. Overclaim scan: absolute language.
bad = []
for phrase in ("proves", "proven", "definitively", "conclusively",
               "for the first time ever", "always", "never generalises"):
    for m in re.finditer(rf"\b{phrase}\b", T, re.I):
        ctx = " ".join(T[max(0, m.start()-60):m.end()+50].split())
        bad.append(f"{phrase}: ...{ctx}...")
if bad:
    findings.append(("absolute-language", "; ".join(bad[:3])))

# 14. Does any number appear in the text that is NOT in the results tables?
nums = set(re.findall(r"\b\d{1,3}\.\d%", T))
print(f"  distinct percentages cited: {len(nums)}")

# 15. The unseen-lab mean is the highest honest number; is it framed safely?
need("unseen-lab-caveated",
     "18.6" in T and ("not reproducible" in T or "swings" in T),
     "68.3% is the most quotable honest figure and the least stable. If a "
     "reader lifts it without the spread, we published a misleading number.")

# 16. Abstract must not contain a figure absent from Results.
abstract = T.split("## Abstract")[1].split("## 1.")[0]
res = T.split("## 3. Results")[1]
orphan = [p for p in re.findall(r"\b\d{1,3}\.\d%", abstract)
          if p not in res and p not in T.split("## 2.")[1].split("## 3.")[0]]
if orphan:
    findings.append(("abstract-orphan-number",
                     f"in abstract but not in methods/results: {orphan}"))

print(f"\n  ADVERSARIAL PASS: {len(findings)} findings\n")
for label, why in findings:
    print(f"    [{label}]")
    print(f"      {why}\n")
if not findings:
    print("    none")
