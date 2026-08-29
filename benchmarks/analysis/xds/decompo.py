"""Decomposition additive du gate, protocole corrige, un seul arbre.

Toutes les cellules viennent de `eegrow_xds`, 27/08 14h22 -> 29/08 16h03, gpu:turing.
Aucun appariement cross-arbre (l'ecart de protocole vaut ~2 pp, cf. JOBS_STATUS).
Unite d'analyse = le sujet tenu a l'ecart ; les seeds partagent donnees et decoupe, donc
elles sont moyennees *dans* le sujet avant toute statistique.
"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
B, RNG = 20000, np.random.default_rng(0)
df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "*.csv"))])
per = (df.groupby(["arm", "model", "align", "subject"]).score.mean()
         .unstack(["arm", "model", "align"]))

def stat(d, label):
    d = np.asarray(d.dropna(), float); n = len(d)
    boot = RNG.choice(d, size=(B, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mde = d.std(ddof=1)*(stats.t.ppf(.975, n-1)+stats.t.ppf(.80, n-1))/np.sqrt(n)
    return dict(contraste=label, delta_pp=100*d.mean(), ic_lo=100*lo, ic_hi=100*hi,
                p=stats.ttest_1samp(d, 0).pvalue, MDE_pp=100*mde,
                gagnants=f"{int((d>0).sum())}/{n}")

def holm(res):
    o = np.argsort(res.p.values); h = np.empty(len(res)); run = 0.
    for r, i in enumerate(o):
        run = max(run, (len(res)-r)*res.p.values[i]); h[i] = min(run, 1.)
    res = res.copy(); res["holm"] = h; return res

P = lambda a, m, al: per[(a, m, al)]
rows = [
  stat(P("pooled","grow_shallow","euclidean") - P("within","bd_shallow","none"),
       "GATE  grow+pooled+EA  vs  bd+within+rien"),
  stat(P("within","bd_shallow","euclidean") - P("within","bd_shallow","none"),
       "  effet principal : EA seule      (bd, within)"),
  stat(P("pooled","bd_shallow","euclidean") - P("within","bd_shallow","euclidean"),
       "  effet principal : pooling seul  (bd, EA)"),
  stat(P("within","grow_shallow","euclidean") - P("within","bd_shallow","euclidean"),
       "  effet principal : croissance    (within, EA)"),
  stat(P("pooled","grow_shallow","euclidean") - P("pooled","bd_shallow","euclidean"),
       "  croissance @ pooled/EA"),
  stat(P("pooled","bd_shallow","none") - P("within","bd_shallow","none"),
       "  pooling seul, sans alignement   (bd)"),
  stat(P("within","bd_shallow","scale") - P("within","bd_shallow","none"),
       "  rescaling seul (bd, within)  -- controle"),
]
print("\n=== decomposition du gate (n=52 sujets, protocole corrige) ===")
print(holm(pd.DataFrame(rows)).round(4).to_string(index=False))

print("\n=== l'EA profite-t-elle PLUS au modele qui croit ? (interaction) ===")
inter = lambda a, b: ((P(a,"grow_shallow","euclidean") - P(a,"grow_shallow",b))
                     -(P(a,"bd_shallow","euclidean")  - P(a,"bd_shallow",b)))
ir = [stat(inter(a, b), f"(grow_EA-grow_{b}) - (bd_EA-bd_{b})  @ {a}")
      for a in ["within","pooled"] for b in ["none","scale"]]
ir.append(stat(inter("pooled","none") - inter("within","none"),
               "difference pooled - within  (le test du mecanisme amplitude)"))
print(holm(pd.DataFrame(ir)).round(4).to_string(index=False))
