# Empreinte du sujet dans les FFN grandis de Conformer (EXPLO_embed_conformer.md, exploratoire)

lee2019_mi, cross_session, n = 54 sujets, hasard = 0.019

fits lus {'tiny': 108, 'random': 108, 'bd': 108}, exclus {}

diagnostics (médianes par fit) : {'tiny': {'n_added': 94.5, 'share_j0': 1.0, 'new_over_old_sqnorm_median': 0.012828935294947438}, 'random': {'n_added': 90.0, 'share_j0': 1.0, 'new_over_old_sqnorm_median': 0.010919158710604018}}

clones random à l'insertion : {'n': 10328, 'other_subject': 0.23024786986831913, 'same_subject': 0.0}

| ensemble | top-1 |
|---|---|
| bd_all | 0.028 |
| bd_matched | 0.016 |
| random_added | 0.000 |
| random_added_ins | 0.009 |
| random_start16 | 0.019 |
| tiny_added | 0.009 |
| tiny_added_ins | 0.009 |
| tiny_start16 | 0.000 |

| contraste (apparié par sujet) | IC95 | MDE |
|---|---|---|
| E0 tiny_added - chance | -0.009 [-0.019, +0.009] | 0.026 |
| E1 tiny_added - bd_matched | -0.006 [-0.020, +0.016] | 0.028 |
| E2 tiny_added - random_added | +0.009 [+0.000, +0.028] | 0.026 |
| random_added - bd_matched | -0.016 [-0.021, -0.011] | 0.008 |
| tiny_added - tiny_start16 | +0.009 [+0.000, +0.028] | 0.026 |
| tiny_added_ins - chance | -0.009 [-0.019, +0.009] | 0.026 |
| random_added_ins - chance | -0.009 [-0.019, +0.009] | 0.026 |
| tiny_added 200 - ins | +0.000 [-0.028, +0.028] | 0.037 |
| random_added 200 - ins | -0.009 [-0.028, +0.000] | 0.026 |

**Lecture (§ 4)** : PAS D'EMPREINTE DÉTECTÉE ; PAS AU-DELÀ DE bd
