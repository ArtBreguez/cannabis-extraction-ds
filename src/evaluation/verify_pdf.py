"""Verify the PDF actually contains the manuscript, then render pages to PNG.

A PDF that opens is not a PDF that is correct. Three failure modes worth
catching before submission: text that silently did not make it in, ASCII
results tables mangled by reflow, and the author block lost so ChemRxiv's
metadata extraction finds nothing.

Uses pypdf for text extraction and pdf2image/pymupdf for rendering, whichever
is available.
"""
import re
import sys
from pathlib import Path

# repo root, resolved from this file so the scripts work in any clone
ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "reports/preprint.pdf"
MD = ROOT / "PREPRINT.md"

try:
    import fitz  # pymupdf
except ImportError:
    print("  pymupdf missing")
    sys.exit(2)

doc = fitz.open(PDF)
print(f"  pages: {doc.page_count}")
print(f"  page size: {doc[0].rect.width:.0f} x {doc[0].rect.height:.0f} pt")

full = "\n".join(p.get_text() for p in doc)
print(f"  extracted characters: {len(full):,}")

# --- does every headline number survive into the PDF? ---
must = ["37,374", "81.0%", "54.0%", "54.6%", "68.3%", "18.6", "56.7%",
        "38.9%", "60.7%", "15.3", "32.3%", "28,927", "77.4%", "99.3%",
        "Arthur Gonçalves Breguez", "0009-0005-8551-731X",
        "arthurbreguez@gmail.com", "Competing interests",
        "Kennard-Stone", "HiddenTerps", "1.50%", "13.5%", "78.4%"]
missing = [m for m in must if m not in full]
print(f"\n  key strings present: {len(must)-len(missing)}/{len(must)}")
if missing:
    print(f"    MISSING: {missing}")

# --- did the ASCII tables keep their alignment? ---
print("\n  === ASCII table integrity ===")
# this row must stay on one line with its columns intact
probes = [
    ("random (optimistic)", "81.0%"),
    ("unseen-producer", "54.0%"),
    ("dual-class + unseen", "54.6%"),
    ("leave-one-variety-out", "38.9%"),
]
for label, val in probes:
    # Several sections mention these names in prose, so check EVERY matching
    # line and pass if any one carries the value on the same line. Checking
    # only the first match reported false splits against the methods list.
    lines = [l for l in full.splitlines() if label in l]
    hit = next((l for l in lines if val in l), None)
    if not lines:
        print(f"    {label:<24} ROW NOT FOUND")
    elif hit:
        print(f"    {label:<24} intact  -> {' '.join(hit.split())[:58]}")
    else:
        print(f"    {label:<24} SPLIT across lines in all "
              f"{len(lines)} occurrences")

# --- heading structure, for metadata extraction ---
print("\n  === structure ===")
for h in ("Abstract", "1. Introduction", "2. Materials and methods",
          "3. Results", "4. Discussion", "5. Conclusion", "References"):
    print(f"    {h:<28} {'found' if h in full else 'MISSING'}")

# --- render the first pages so the layout can be inspected ---
out = ROOT / "reports/pages"
out.mkdir(exist_ok=True)
for i in range(min(3, doc.page_count)):
    pix = doc[i].get_pixmap(dpi=110)
    p = out / f"page{i+1}.png"
    pix.save(p)
    print(f"  rendered {p.name} ({p.stat().st_size//1024} KB)")
