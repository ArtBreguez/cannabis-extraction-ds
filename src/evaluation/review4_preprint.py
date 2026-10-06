"""Fourth pass: do the manuscript's claims about ITSELF hold?

Earlier passes checked figures (audit), referee objections (review1), internal
consistency (review2), and citations (review3). None verified the claims the
paper makes about its own artefacts: that a named script exists, that it
recomputes what the text says it recomputes, that the directory layout printed
in Data availability matches reality, and that the public URL resolves.

A paper whose thesis is reproducibility cannot afford a broken self-reference.
"""
import re
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = (ROOT / "PREPRINT.md").read_text()
fails = []


def check(label, ok, detail=""):
    print(f"    {label:<46} {'OK' if ok else 'FAIL'}  {detail}")
    if not ok:
        fails.append(label)


print("  === 1. every path the manuscript names must exist ===")
for m in sorted(set(re.findall(r"`(src/[\w/]+\.py)`", T))):
    check(m, (ROOT / m).exists())
for m in sorted(set(re.findall(r"^(src/[\w]+/|docs/evidence/)\s",
                              T, re.M))):
    check(m, (ROOT / m.strip()).is_dir())

print("\n  === 2. do the named scripts do what the text claims? ===")
rederive = ROOT / "src/evaluation/rederive_paper_numbers.py"
src = rederive.read_text() if rederive.exists() else ""
check("rederive reads the labelled dataset, not a doc",
      "master.csv" in src and "PHASE4_RESULTS" not in src)
audit = (ROOT / "src/evaluation/audit_preprint.py")
asrc = audit.read_text() if audit.exists() else ""
check("audit fails loudly on disagreement",
      "failed" in asrc and ("sys.exit" in asrc or "SystemExit" in asrc
                            or "FAIL" in asrc))

print("\n  === 3. the public URL in the text actually resolves ===")
urls = re.findall(r"https://github\.com/[\w.-]+/[\w.-]+", T)
for u in sorted(set(urls)):
    try:
        req = urllib.request.Request(u, method="HEAD",
                                     headers={"User-Agent": "curl/8"})
        code = urllib.request.urlopen(req, timeout=25).status
    except Exception as e:
        code = getattr(e, "code", str(e))
    check(u, code == 200, f"HTTP {code}")

print("\n  === 4. is the repo state consistent with the claim? ===")


def sh(c):
    return subprocess.run(c, shell=True, cwd=ROOT, capture_output=True,
                          text=True, timeout=300).stdout.strip()


check("repository is public", sh("gh repo view --json isPrivate -q .isPrivate")
      == "false")
check("LICENSE is tracked", "LICENSE" in sh("git ls-files"))
check("data/ is NOT tracked", not sh("git ls-files -- data/"))
check("no absolute local path in tracked code",
      not sh("git grep -lI '/home/arthur' -- 'src/*.py'"))
check("working tree committed", not sh("git status --porcelain"))

print("\n  === 5. abstract standalone: does it overstate? ===")
abstract = T.split("## Abstract")[1].split("**Keywords")[0]
check("abstract states the chance level", "50%" in abstract
      or "chance" in abstract.lower())
check("abstract does not call others' numbers inflated",
      not re.search(r"(?i)(their|published).{0,30}(inflat|wrong|invalid)",
                    abstract))
check("abstract declares the negative result plainly",
      re.search(r"(?i)not (recoverable|identifiable|readable)", abstract)
      is not None)
check("abstract carries no claim absent from Results",
      all(p in T.split("## 3. Results")[1]
          for p in set(re.findall(r"\b\d{1,3}\.\d%", abstract))))

print("\n  === 6. the one thing a referee will test first ===")
# The headline is a 27-point drop. Confirm both ends and the arithmetic.
res = T.split("## 3. Results")[1]
has81 = "81.0%" in res
has54 = "54.0%" in res
drop = re.search(r"27(\.0)?\s*(point|pt)", T)
check("both endpoints present in Results", has81 and has54)
check("the stated drop equals 81.0 - 54.0", bool(drop)
      and abs(81.0 - 54.0 - 27.0) < 1e-9)

print(f"\n  {'ALL CLEAR' if not fails else 'FAILURES: ' + str(fails)}")
