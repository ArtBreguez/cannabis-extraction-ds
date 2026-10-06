"""Figure 1, drawn from the committed evidence logs and nothing else.

Every plotted value is parsed out of docs/evidence/*.txt, so the figure cannot
drift from the numbers the manuscript states: if a log changes, rerunning this
script redraws the figure, and audit_preprint.py checks that the plotted
values are the logged ones. Output is a hand-written SVG (no plotting library
is needed to reproduce it).

  (a) capacity: the random split climbs with boosting iterations, the
      producer-held-out score does not; a logistic regression lands on the
      same held-out score from a much lower random split
  (b) twenty producer-to-fold assignments for the two producer-held-out
      schemes, with the producer-cluster bootstrap interval of the pooled
      score, against the chance level

Output: reports/figures/fig1_validation_gap.svg and a JSON of plotted values
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "docs/evidence"
OUT = ROOT / "reports/figures/fig1_validation_gap.svg"
VALS = ROOT / "reports/figures/fig1_validation_gap.json"

BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1 and 2, validated
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e3e2de", "#b9b8b2"
FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"


def parse():
    b = (EV / "review10b_robustness.txt").read_text()
    a = (EV / "review10a_grouped.txt").read_text()
    r7 = (EV / "review7_statistical.txt").read_text()
    r9 = (EV / "review9_checks.txt").read_text()
    cap = [(int(m), float(r), float(g)) for m, r, g in re.findall(
        r"max_iter=(\d+)\s+random ([\d.]+)%.*?unseen-producer ([\d.]+)%", b)]
    lr = float(re.search(r"random, producer rows\s+([\d.]+)%", b).group(1))
    lg = float(re.search(r"unseen-producer\s+([\d.]+)% \(\+/-[\d.]+\)\s+same-sample gap", b).group(1))
    parts = {}
    for name in ("unseen-producer", "dual-class+unseen"):
        blk = a.split(f"=== {name}:")[1].split("\n===")[0]
        vals = [float(v) for v in re.search(r"values: \[([\d., ]+)\]", blk).group(1).split(",")]
        lo, hi = map(float, re.search(r"95% interval \[([\d.]+), ([\d.]+)\]", blk).groups())
        pooled = float(re.search(r"pooled balanced_acc ([\d.]+)%", blk).group(1))
        parts[name] = {"values": vals, "boot": [lo, hi], "pooled": pooled}
    parts["unseen-producer"]["random"] = float(
        re.search(r"random, producer rows only\s+([\d.]+)%", r7).group(1))
    parts["dual-class+unseen"]["random"] = float(
        re.search(r"random 5-fold ([\d.]+)%", r9).group(1))
    return {"capacity": cap, "logistic": [lr, lg], "partitions": parts}


def main() -> int:
    d = parse()
    W, H = 720, 320
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="{FONT}" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']

    def text(x, y, t, size=11, fill=INK, anchor="start", weight="normal"):
        s.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                 f'text-anchor="{anchor}" font-weight="{weight}">{t}</text>')

    # ---------------- panel (a): capacity ----------------
    ax0, ax1, ay0, ay1 = 48, 318, 74, 272          # plot box
    ymin, ymax = 45.0, 85.0
    cats = ["logistic"] + [str(c[0]) for c in d["capacity"]]
    xs = [ax0 + 34 + i * (ax1 - ax0 - 68) / (len(cats) - 1) for i in range(len(cats))]

    def Y(v):
        return ay1 - (v - ymin) / (ymax - ymin) * (ay1 - ay0)

    text(ax0 - 34, 22, "a", 13, weight="bold")
    text(ax0 - 18, 22, "Score against model capacity", 10.5)
    # legend
    for i, (c, lab) in enumerate(((BLUE, "random split"), (ORANGE, "producer held out"))):
        lx = ax0 - 34 + i * 118
        s.append(f'<line x1="{lx}" y1="40" x2="{lx+18}" y2="40" stroke="{c}" stroke-width="2"/>')
        s.append(f'<circle cx="{lx+9}" cy="40" r="4" fill="{c}" stroke="#fff" stroke-width="2"/>')
        text(lx + 24, 43.5, lab, 10, MUTED)
    for v in (50, 60, 70, 80):
        s.append(f'<line x1="{ax0}" y1="{Y(v):.1f}" x2="{ax1}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        text(ax0 - 6, Y(v) + 3.5, f"{v}", 10, MUTED, "end")
    s.append(f'<line x1="{ax0}" y1="{Y(50):.1f}" x2="{ax1}" y2="{Y(50):.1f}" stroke="{BAND}" stroke-width="1"/>')
    text(ax1 - 2, Y(50) + 12, "chance", 9.5, MUTED, "end")
    text(ax0 - 34, ay0 - 9, "balanced accuracy (%)", 9.5, MUTED)
    for x, c in zip(xs, cats):
        text(x, ay1 + 15, c, 10, MUTED, "middle")
    text((xs[1] + xs[-1]) / 2, ay1 + 30, "gradient boosting, iterations", 9.5, MUTED, "middle")
    text(xs[0], ay1 + 30, "regression", 9.5, MUTED, "middle")
    s.append(f'<line x1="{(xs[0]+xs[1])/2:.1f}" y1="{ay0}" x2="{(xs[0]+xs[1])/2:.1f}" y2="{ay1}" stroke="{GRID}" stroke-width="1"/>')
    rnd = [c[1] for c in d["capacity"]]
    grp = [c[2] for c in d["capacity"]]
    for vals, col in ((rnd, BLUE), (grp, ORANGE)):
        pts = " ".join(f"{x:.1f},{Y(v):.1f}" for x, v in zip(xs[1:], vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, v in zip(xs[1:], vals):
            s.append(f'<circle cx="{x:.1f}" cy="{Y(v):.1f}" r="4" fill="{col}" stroke="#fff" stroke-width="2"/>')
    for v, col in zip(d["logistic"], (BLUE, ORANGE)):
        s.append(f'<circle cx="{xs[0]:.1f}" cy="{Y(v):.1f}" r="4" fill="{col}" stroke="#fff" stroke-width="2"/>')
    # selective direct labels: the two ends and the logistic pair
    text(xs[-1] + 8, Y(rnd[-1]) + 3.5, f"{rnd[-1]:.1f}", 10)
    text(xs[3], Y(rnd[2]) - 9, f"{rnd[2]:.1f}", 10, anchor="middle")
    text(xs[-1] + 8, Y(grp[-1]) - 6, f"{grp[-1]:.1f}", 10)
    text(xs[1] - 8, Y(rnd[0]) - 6, f"{rnd[0]:.1f}", 10, anchor="end")
    text(xs[1] - 8, Y(grp[0]) - 6, f"{grp[0]:.1f}", 10, anchor="end")
    text(xs[0], Y(d["logistic"][0]) - 9, f"{d['logistic'][0]:.1f}", 10, anchor="middle")
    text(xs[0], Y(d["logistic"][1]) - 9, f"{d['logistic'][1]:.1f}", 10, anchor="middle")

    # ---------------- panel (b): partitions ----------------
    bx0, bx1 = 494, 704
    xmin, xmax = 44.0, 62.0

    def X(v):
        return bx0 + (v - xmin) / (xmax - xmin) * (bx1 - bx0)

    lab_x = 372
    text(lab_x, 22, "b", 13, weight="bold")
    text(lab_x + 16, 22, "Producer held out: twenty fold assignments", 10.5)
    s.append(f'<circle cx="{lab_x+5}" cy="40" r="3.2" fill="{ORANGE}" stroke="#fff" stroke-width="1.3"/>')
    text(lab_x + 13, 43.5, "one assignment", 10, MUTED)
    s.append(f'<line x1="{lab_x+112}" y1="40" x2="{lab_x+132}" y2="40" stroke="{BAND}" stroke-width="5" stroke-linecap="round"/>')
    s.append(f'<line x1="{lab_x+122}" y1="34" x2="{lab_x+122}" y2="46" stroke="{INK}" stroke-width="2"/>')
    text(lab_x + 140, 43.5, "pooled score, producer bootstrap 95%", 10, MUTED)
    for v in (45, 50, 55, 60):
        s.append(f'<line x1="{X(v):.1f}" y1="{ay0}" x2="{X(v):.1f}" y2="{ay1}" stroke="{GRID}" stroke-width="1"/>')
        text(X(v), ay1 + 15, f"{v}", 10, MUTED, "middle")
    s.append(f'<line x1="{X(50):.1f}" y1="{ay0}" x2="{X(50):.1f}" y2="{ay1}" stroke="{BAND}" stroke-width="1"/>')
    text(X(50) - 4, ay0 + 10, "chance", 9.5, MUTED, "end")
    text((bx0 + bx1) / 2, ay1 + 30, "balanced accuracy (%)", 9.5, MUTED, "middle")
    rows = (("unseen-producer", "all producers", "96 producers"),
            ("dual-class+unseen", "dual-class only", "53 producers"))
    for k, (key, lab, sub) in enumerate(rows):
        cy = ay0 + 44 + k * 100
        p = d["partitions"][key]
        lo, hi = p["boot"]
        text(lab_x, cy - 6, lab, 10)
        text(lab_x, cy + 7, sub, 9.5, MUTED)
        text(lab_x, cy + 22, f"{min(p['values']):.1f} to {max(p['values']):.1f}", 9.5, MUTED)
        text(lab_x, cy + 35, f"pooled {p['pooled']:.1f} ({lo:.1f} to {hi:.1f})", 9.5, MUTED)
        placed = []
        for v in sorted(p["values"]):
            x, lvl = X(v), 0
            while any(abs(x - px) < 7.0 and lvl == pl for px, pl in placed):
                lvl += 1
            placed.append((x, lvl))
            s.append(f'<circle cx="{x:.1f}" cy="{cy + 12 - 7.5 * lvl:.1f}" r="3.2" fill="{ORANGE}" '
                     f'stroke="#fff" stroke-width="1.3"/>')
        s.append(f'<line x1="{X(lo):.1f}" y1="{cy+30}" x2="{X(hi):.1f}" y2="{cy+30}" '
                 f'stroke="{BAND}" stroke-width="5" stroke-linecap="round"/>')
        s.append(f'<line x1="{X(p["pooled"]):.1f}" y1="{cy+24}" x2="{X(p["pooled"]):.1f}" y2="{cy+36}" '
                 f'stroke="{INK}" stroke-width="2"/>')
    s.append("</svg>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(s) + "\n", encoding="utf-8")
    VALS.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} and {VALS.relative_to(ROOT)}")
    print(json.dumps(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
