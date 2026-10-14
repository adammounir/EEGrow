"""Calibrate the gradient-spike refusal threshold on the previous campaign (13/10/2026).

Run on Margaret, read-only, in /scratch/amounir/results_final (167 076 fit records, all
9 arms, every dataset/eval). Per fit: the largest per-batch gradient norm over the whole
fit (`grad_norm_max`, recorded by GradientNorm), histogrammed in quarter decades.

Result (13/10):
  * healthy fits top out at 10^3.75-10^4 (grow_sccnet, the arm with the largest normal
    norms); one isolated grow_sccnet fit at 5.8e4;
  * NO fit at all between 5.8e4 and 5.6e6 (two decades empty);
  * spikes: 222 fits from 5.6e6 to 1e16, all shallow/sccnet arms, 0 on bd_* and the
    deep family.
  => threshold 1e6 on grad_norm_max: x17 above the largest healthy value, x5.6 below
  the smallest spike, with zero fits in between on 167 076.
  The ratio to the fit's own median grad norm is NOT usable: memorisation collapses the
  late norm, so healthy fits reach ratios of 1e4-1e5 (bd_shallow p99 = 10^4.1).
  The 04/10 count "692 fits > 1e3" mixed in grow_sccnet's normal range (471 fits in
  1e3-1e4); the true spike count is 222.

    python3 spike_threshold.py      # from /scratch/amounir/results_final
"""
import glob
import json
import math
from collections import Counter, defaultdict

fine = defaultdict(Counter)
for f in glob.glob("*/*/*__fits.jsonl"):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        gx = [h["grad_norm_max"] for h in r["history"] if h.get("grad_norm_max")]
        if not gx:
            continue
        model = r.get("model") or f.split("/")[2].split("__")[0]
        fine[math.floor(4 * math.log10(max(gx))) / 4][model] += 1
for b in sorted(fine):
    print(f"10^{b:5.2f}", sum(fine[b].values()), dict(fine[b]))
