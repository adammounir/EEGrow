import sys, numpy as np, pandas as pd
from pathlib import Path
A=Path("/Users/adammounir/Desktop/Inria/Exploration/eegrow/benchmarks/analysis"); sys.path.insert(0,str(A))
import donor_matrix as dm, donor_predictor as dp
rng=np.random.default_rng(20261006)
def frame(ds):
    s=pd.read_csv(f"subj_{ds}.csv"); rk=pd.read_csv(A/f"ranking_{ds}.csv")[["subject","acc_probe"]]
    q,_=dm.donor_quality(dm.load_matrix(A/f"dxr_{ds}"),dm.METRIC)
    return s.merge(rk,on="subject").merge(q[["donor","q_cent"]],left_on="subject",right_on="donor").dropna(subset=["orig","late","cv","q_cent","acc_probe"])
def pooled(fr,k):
    z=[];w=[]
    for m in fr:
        z.append(np.arctanh(dp.partial_spearman(m[k].values,m.q_cent.values,m.acc_probe.values))); w.append(len(m)-4)
    return np.tanh(np.average(z,weights=w))
import contextlib,io
for name,dss in (("étage C",["physionetmi","lee2019_mi","cho2017"]),("frais",["dreyer2023a","liu2024"])):
    with contextlib.redirect_stdout(io.StringIO()): fr=[frame(d) for d in dss]
    pt={k:pooled(fr,k) for k in ("orig","late","cv")}
    B={k:[] for k in ("late","cv")}
    for _ in range(4000):
        bf=[m.iloc[rng.integers(0,len(m),len(m))] for m in fr]
        o=pooled(bf,"orig")
        for k in B: B[k].append(pooled(bf,k)-o)
    print(name, {k:round(v,3) for k,v in pt.items()})
    for k,v in B.items():
        lo,hi=np.percentile(v,[2.5,97.5]); print(f"   {k}-orig {pt[k]-pt['orig']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
