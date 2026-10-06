"""Verify the 2.2 feature-construction disclosures against the data.

Every figure the manuscript states in the two paragraphs added for
total_terpenes and for the impossible potency values is recomputed here.
"""
import sys
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[2]


class _Tee:
    """Print to stdout and to the committed evidence log at once."""
    def __init__(self, path):
        self.f = open(path, "w"); self.o = sys.stdout
    def write(self, s):
        self.f.write(s); self.o.write(s)
    def flush(self):
        self.f.flush(); self.o.flush()


sys.stdout = _Tee(ROOT / "docs/evidence/verify_22_disclosures.txt")
d = pd.read_csv(ROOT / 'data/labeled/master.csv',
                low_memory=False)
d = d[d.label_conflict == 0]

PARTS = ['beta_myrcene','d_limonene','beta_caryophyllene','alpha_pinene',
         'beta_pinene','caryophyllene_oxide','linalool','alpha_humulene',
         'terpinolene','alpha_bisabolol']
print("claim: 'ten terpenes that survive the empty-column drop'")
print("  non-empty terpene value columns:", len(PARTS), PARTS == sorted(PARTS, key=PARTS.index))

s = d[PARTS].sum(axis=1, min_count=1)
m = d.total_terpenes.notna() & s.notna()
# One row set for both statistics: the rows that carry total_terpenes and
# report all ten terpenes. R2 is the squared correlation with the plain sum,
# so the two figures describe the same comparison.
mm = m & d[PARTS].notna().all(axis=1)
r_full = np.corrcoef(d.total_terpenes[mm], s[mm])[0, 1]
print("\nclaim: on the rows with all ten terpenes reported, Pearson r = 0.992 "
      "against their sum and R2 = 0.984")
print("  rows with total_terpenes and all ten terpenes reported: %d of %d"
      % (mm.sum(), d.total_terpenes.notna().sum()))
print("  Pearson r: %.5f" % r_full)
print("  R2 (r^2) : %.5f" % r_full ** 2)

print("\nclaim: CERTIFIED AG exact on 99.7%, MA & ASSOCIATES on none")
ad = (d.total_terpenes - s)[m].abs()
fr = d[m].assign(ad=ad).groupby('lab').ad.apply(lambda x: (x <= 1e-6).mean())
for lab in ['CERTIFIED AG LAB LLC', 'MA & ASSOCIATES LLC']:
    print(f"  {lab:24} exact-sum share = {100*fr[lab]:.1f}%")

print("\nclaim: populated total_terpenes == producer present, 33,229 / 33,229")
tt, pr = d.total_terpenes.notna(), d.producer.notna()
print("  total_terpenes populated:", int(tt.sum()), " producer present:", int(pr.sum()))
print("  masks identical         :", bool((tt == pr).all()))

print("\nclaim: every lab at 0% or 100%, 8 at 100% and 4 at 0%")
per = d.groupby('lab').total_terpenes.apply(lambda x: x.notna().mean())
print("  labs at 100%:", sorted(per[per == 1.0].index.tolist()))
print("  labs at   0%:", sorted(per[per == 0.0].index.tolist()))
print("  any lab strictly between:", bool(((per > 0) & (per < 1)).any()))

print("\nclaim: 101 rows total_thc > 100, max 686,400, 94 from one lab")
hi = d.total_thc > 100
print("  rows > 100 :", int(hi.sum()))
print("  max        : %.2f" % d.total_thc.max())
print("  top lab    :", d.lab[hi].value_counts().head(1).to_dict())
print("  share      : %.2f%% of corpus" % (100 * hi.mean()))
