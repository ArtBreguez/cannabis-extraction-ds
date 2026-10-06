"""Third pass: citation integrity and submission readiness.

The audit checks figures, review1 checks referee objections, review2 checked
internal consistency. None of them checked the reference list, which is where
a manuscript most often embarrasses its author: a citation marker with no
entry, an entry nobody cites, a DOI that does not resolve, or a claim
attributed to the wrong paper.

Also checks the ChemRxiv submission gates, since "ready to publish" means
ready for THAT screening, not just internally coherent.
"""
import re
import subprocess
from pathlib import Path

# repo root, resolved from this file so the scripts work in any clone
ROOT = Path(__file__).resolve().parents[2]
T = (ROOT / "PREPRINT.md").read_text()

print("  === 1. every [n] marker has an entry, every entry is cited ===")
refs_block = T.split("## References")[1]
entries = set(int(m) for m in re.findall(r"^\[(\d+)\]", refs_block, re.M))
body = T.split("## References")[0]
# Markers appear both singly as [8] and in lists as [4, 5, 6, 7]; the naive
# pattern missed every list member and falsely reported five uncited entries.
cited = set()
for group in re.findall(r"\[([\d,\s]+)\]", body):
    for num in re.findall(r"\d+", group):
        cited.add(int(num))
print(f"    entries in list : {sorted(entries)}")
print(f"    cited in body   : {sorted(cited)}")
orphan_entry = entries - cited
dangling = cited - entries
if orphan_entry:
    print(f"    UNCITED ENTRIES : {sorted(orphan_entry)}  <- remove or cite")
if dangling:
    print(f"    DANGLING MARKERS: {sorted(dangling)}  <- no entry exists")
if not orphan_entry and not dangling:
    print("    -> consistent")

print("\n  === 2. are the DOIs resolvable? ===")
dois = re.findall(r"doi:(10\.\d{4,}/[^\s.,]+(?:\.[^\s.,]+)*)", refs_block)
for d in sorted(set(dois)):
    r = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         "-L", f"https://doi.org/{d}", "--max-time", "25"],
        capture_output=True, text=True, timeout=60)
    code = r.stdout.strip()
    ok = code in ("200", "302", "403")  # 403 = publisher blocks bots
    print(f"    {d:<40} HTTP {code}  {'resolves' if ok else 'CHECK'}")

print("\n  === 3. claim-to-citation mapping ===")
# Each specific numeric claim about another paper must sit next to its marker.
claims = [
    ("95.2%", "1", "SVM accuracy on resin type"),
    ("100% prediction accuracy", "2", "PLS-DA chemovar class"),
    ("88 genotypes cloned in triplicate", "2", "the clone structure"),
    ("149/40", "1", "the class split behind 95.2%"),
    ("Kennard-Stone", "8", "the leakage mechanism"),
]
for phrase, ref, what in claims:
    idx = T.find(phrase)
    if idx < 0:
        print(f"    {phrase[:30]:<32} PHRASE ABSENT")
        continue
    window = T[max(0, idx - 320): idx + 320]
    found = f"[{ref}]" in window
    print(f"    {phrase[:30]:<32} ref [{ref}] "
          f"{'present nearby' if found else 'NOT NEAR -> attribution unclear'}"
          f"   ({what})")

print("\n  === 4. ChemRxiv submission gates ===")
gates = [
    ("English only", not re.search(r"\b(medimos|resultado|porque|análise)\b",
                                   T, re.I)),
    ("ORCID present", "0009-0005-8551-731X" in T),
    ("full legal name", "Arthur Gonçalves Breguez" in T),
    ("competing interests declared", "## Competing interests" in T
     or "Competing interests" in T),
    ("no clinical data", "patient" not in T.lower()
     and "clinical trial" not in T.lower()),
    ("data availability stated", "Data and code availability" in T),
    ("not previously published", "Preprint." in T),
    ("abstract present", "## Abstract" in T),
]
for label, ok in gates:
    print(f"    {label:<32} {'OK' if ok else 'FAIL'}")

print("\n  === 5. is the code-availability claim honest and checkable? ===")
repo_mentions = len(re.findall(r"cannabis-extraction-ds", T))
has_url = bool(re.search(r"github\.com/\S+cannabis-extraction-ds", T))
promises_later = bool(re.search(
    r"will be published alongside|DOI will be added|available\s+from the author",
    T))
print(f"    repo named {repo_mentions} times")
print(f"    resolvable URL given: {has_url}")
print(f"    publication promised with a route meanwhile: {promises_later}")
if has_url:
    print("    -> fully verifiable")
elif promises_later:
    print("    -> acceptable: no URL exists yet, and the text says so rather")
    print("       than implying a repository a reader could already reach.")
    print("       MUST be replaced with the real URL/DOI before posting.")
else:
    print("    FAIL: cites a repository with no URL and no explanation.")

print("\n  === 6. numbers in the abstract vs the results ===")
abstract = T.split("## Abstract")[1].split("**Keywords")[0]
res = T.split("## 3. Results")[1].split("## 4.")[0]
meth = T.split("## 2.")[1].split("## 3.")[0]
intro = T.split("## 1.")[1].split("## 2.")[0]
for p in sorted(set(re.findall(r"\b\d{1,3}\.\d%", abstract))):
    where = [n for n, sec in (("results", res), ("methods", meth),
                              ("intro", intro)) if p in sec]
    print(f"    {p:<8} appears in: {', '.join(where) or 'NOWHERE ELSE'}")
