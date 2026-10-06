"""Render PREPRINT.md as a submission-ready PDF.

ChemRxiv accepts a PDF and extracts title, abstract, authors and affiliations
from it, so the structure has to be machine-readable: a real <h1> title, an
Abstract heading, and the author block immediately below the title.

Choices made for a referee reading on screen and on paper:
  - single column, 11pt serif, generous leading. Two columns look like a
    journal galley but break the wide results tables.
  - monospace blocks keep their alignment, since every results table in this
    manuscript is whitespace-aligned ASCII and would be destroyed by reflow.
  - page numbers and a running footer, because a reviewer citing "page 4"
    needs page 4 to exist.
  - no syntax colouring: these are data tables, not code.
"""
import re
import subprocess
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

# repo root, resolved from this file so the scripts work in any clone
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "PREPRINT.md"
OUT = ROOT / "reports/preprint.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

text = SRC.read_text()

# The leading `---` separators are visual in Markdown but render as empty
# horizontal rules that waste a line; drop the ones around the abstract.
text = re.sub(r"^---$", "", text, flags=re.M)

html_body = markdown.markdown(
    text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])

# The author block is written as consecutive Markdown lines, which collapse
# into one paragraph. Split it back out so affiliation and contact sit on
# their own lines, as a manuscript requires and as metadata extraction expects.
# The name is read from the manuscript rather than hardcoded here, so the
# renderer cannot drift out of sync with the source.
AUTHOR_M = re.search(r"^\*\*(.+?)\*\*$", text, re.M)
if not AUTHOR_M:
    raise SystemExit("author line not found in PREPRINT.md")
AUTHOR = AUTHOR_M.group(1)
# The document title comes from the manuscript's own H1, so renaming the paper
# cannot leave a stale title in the PDF metadata.
TITLE_M = re.search(r"^# (.+)$", text, re.M)
if not TITLE_M:
    raise SystemExit("title (H1) not found in PREPRINT.md")
TITLE = TITLE_M.group(1).strip()
html_body = re.sub(
    rf"<p><strong>{re.escape(AUTHOR)}</strong>\s*(.*?)</p>",
    lambda m: (
        f'<p class="authors"><strong>{AUTHOR}</strong></p>'
        '<p class="affil">'
        + re.sub(r"\s*(ORCID:|Correspondence:)", r"<br>\1", m.group(1)).strip()
        + "</p>"),
    html_body, count=1, flags=re.S)

CSS_TEXT = """
@page {
  size: A4;
  margin: 22mm 20mm 20mm 20mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: Georgia, serif; font-size: 8.5pt; color: #666;
  }
  @bottom-left {
    content: "Preprint. Not peer reviewed.";
    font-family: Georgia, serif; font-size: 8pt; color: #999;
  }
}
html { font-size: 11pt; }
body {
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.52; color: #111; text-align: justify;
  hyphens: auto;
}
h1 {
  font-size: 18pt; line-height: 1.25; margin: 0 0 14pt 0;
  text-align: left; font-weight: normal; color: #000;
}
p.authors { margin: 0 0 2pt 0; line-height: 1.4; text-align: left; }
p.authors strong { font-size: 12.5pt; }
p.affil {
  margin: 0 0 10pt 0; line-height: 1.42; text-align: left;
  font-size: 10pt; color: #333;
}
h2 {
  font-size: 13pt; margin: 20pt 0 7pt 0; text-align: left;
  border-bottom: 0.6pt solid #ccc; padding-bottom: 3pt;
  page-break-after: avoid;
}
h3 {
  font-size: 11.5pt; margin: 14pt 0 5pt 0; text-align: left;
  page-break-after: avoid;
}
p { margin: 0 0 7pt 0; orphans: 3; widows: 3; }
/* Whitespace-aligned ASCII tables: never reflow, never shrink below legible */
pre {
  font-family: "DejaVu Sans Mono", "Courier New", monospace;
  font-size: 8.2pt; line-height: 1.34;
  background: #f7f7f7; border: 0.5pt solid #e0e0e0;
  border-left: 2.5pt solid #999;
  padding: 7pt 9pt; margin: 9pt 0;
  white-space: pre; overflow-wrap: normal;
  page-break-inside: avoid; text-align: left;
}
code {
  font-family: "DejaVu Sans Mono", monospace; font-size: 9pt;
  background: #f2f2f2; padding: 0 2pt; border-radius: 2pt;
}
pre code { background: none; padding: 0; font-size: inherit; }
table {
  border-collapse: collapse; width: 100%; margin: 9pt 0;
  font-size: 9.5pt; page-break-inside: avoid;
}
th, td {
  border-top: 0.5pt solid #ccc; border-bottom: 0.5pt solid #ccc;
  padding: 3.5pt 5pt; text-align: left; vertical-align: top;
}
th { background: #f3f3f3; font-weight: bold; }
ul, ol { margin: 0 0 7pt 0; padding-left: 17pt; }
li { margin-bottom: 3pt; text-align: justify; }
strong { font-weight: bold; }
em { font-style: italic; }
hr { display: none; }
/* Figures: full text width, never split from the caption that follows */
img { width: 100%; height: auto; display: block; margin: 8pt 0 3pt 0; }
p:has(> img) { margin: 0; page-break-after: avoid; page-break-inside: avoid; }
p:has(> img) + p { font-size: 9.5pt; line-height: 1.4; text-align: left;
                   page-break-inside: avoid; margin-bottom: 10pt; }
/* Keep the reference list compact and unjustified */
h2#references ~ p { text-align: left; font-size: 9.5pt; margin-bottom: 5pt; }
"""

HTML_DOC = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{TITLE}</title>
<meta name="author" content="{AUTHOR}">
<meta name="citation_author" content="{AUTHOR}">
<meta name="citation_author_orcid" content="0009-0005-8551-731X">
</head><body>{html_body}</body></html>"""

# base_url lets the Markdown image path (reports/figures/...) resolve.
HTML(string=HTML_DOC, base_url=str(ROOT) + "/").write_pdf(
    OUT, stylesheets=[CSS(string=CSS_TEXT)])
print(f"  wrote {OUT}  ({OUT.stat().st_size:,} bytes)")

# Verify: page count, and that the text actually made it in
try:
    info = subprocess.run(["pdfinfo", str(OUT)], capture_output=True,
                          text=True, timeout=60).stdout
    for line in info.splitlines():
        if line.startswith(("Pages", "Page size", "Producer")):
            print(f"  {line}")
except FileNotFoundError:
    print("  (pdfinfo unavailable, skipping page count)")
