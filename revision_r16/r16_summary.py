#!/usr/bin/env python3
"""r16_summary.py <results_dir> — medians over seeds per scenario x algorithm (+ E2 reference from R1.3)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
d = Path(sys.argv[1]); rows = {}
for f in sorted(d.glob("*/*/seed_*.json")):
    r = json.loads(f.read_text()); rows.setdefault((r["scenario"], r["algo"]), []).append(r)
med = lambda v: float(np.median(v))
print("scen  algo       n  Gini   fair  epGini  top  ho/ep  ep_w_ho  tasks/ep  coll   moves  reward   p_fair(vs nash) p_Gini")
for scn in ["S1", "S2", "S3"]:
    nash = rows.get((scn, "nash"), [])
    for algo in ["nash", "nash_util", "qmix", "vdn"]:
        R = rows.get((scn, algo), [])
        if not R: continue
        F = [r["frozen"] for r in R]; g = [f["gini"] for f in F]; fair = sum(x < 0.15 for x in g)
        p1 = p2 = ""
        if algo != "nash" and nash:
            gn = [r["frozen"]["gini"] for r in nash]; fn = sum(x < 0.15 for x in gn)
            p1 = f"{fisher_exact([[fn, len(gn) - fn], [fair, len(g) - fair]]).pvalue:.4f}"; p2 = f"{mannwhitneyu(gn, g).pvalue:.4f}"
        sh = [f["top_task_share"] for f in F if f["top_task_share"] is not None]
        print(f"{scn:5s} {algo:9s} {len(R):2d} {med(g):.3f} {fair:3d}/{len(R)} {med([f['ep_gini_median'] for f in F]):.3f}  "
              f"{med(sh):.2f}   {med([f['handovers_per_ep'] for f in F]):5.2f}  {med([f['episodes_with_handover'] for f in F]):5.0f}  "
              f"{med([f['tasks_per_ep'] for f in F]):7.1f}  {med([f['collisions_per_ep'] for f in F]):5.2f}  "
              f"{med([f['moves_per_ep'] for f in F]):5.1f}  {med([f['total_reward_per_ep'] for f in F]):6.0f}   {p1:>8s} {p2:>8s}")
