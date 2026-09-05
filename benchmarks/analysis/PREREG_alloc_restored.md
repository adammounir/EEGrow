# Pré-registration — `restored_probe` comme règle d'allocation, et l'écran hors ligne

Écrite le **04/09/2026**, avant le premier fit de la campagne `dalloc_restored`.
Classements et matrices D×R vérifiés sur disque juste avant (7 datasets, 24 160 paires).

## D'où ça vient

Le face-à-face de la campagne `dalloc` a tranché contre le signal de croissance :
`acc_probe` bat `params_probe` 12/12 sur cho2017, égalité sur physionetmi, et
`params_probe` ne gagne un peu que sur lee2019_mi. Voir [[allocation-continuum]].

Deux choses sont apparues en cherchant pourquoi, et elles motivent cette préreg.

**1. La mesure pré-enregistrée n'a jamais été testée en intervention.** H1 de
[[claim2-fresh-replication]] déclare `restored_epoch` comme prédicteur primaire. La
campagne `dalloc` a balayé `acc_probe` et `params_probe`. Le tuple `MEASURES` de
`donor_allocate.py` contenait bien `restored_epoch`, mais sous le nom de la colonne brute
de `gd_fits` alors que le classement écrit `restored_probe` : la mesure était déclarée et
**inatteignable**, tout lancement se serait fait refuser par le garde-fou. Corrigé le
04/09 dans le même commit que cette préreg.

**2. Un écran hors ligne existe et il est gratuit.** Les matrices D×R contiennent le score
de transfert mesuré de chaque paire (donneur, receveur). Noter une règle de tri sans
aucun GPU, c'est prendre les `k` premiers donneurs qu'elle désigne et lire leur transfert
moyen. C'est l'analogue direct de ce que l'allocation exploite — la concentration ne se
sert que du haut du classement, pas du Spearman sur tout le vivier.

L'écran hors ligne, top-5, delta `restored_probe` − `acc_probe`, apparié par receveur :

| dataset | delta top-5 | IC95 |
|---|---|---|
| cho2017 | **+0.0387** | [+0.0237, +0.0551] |
| lee2019_mi | **+0.0143** | [+0.0046, +0.0237] |
| physionetmi | **−0.0286** | [−0.0369, −0.0204] |

## Ce qui est testé

**H_A (primaire).** À budget d'essais fixe, allouer par `restored_probe` plutôt que par
`acc_probe` change le score du receveur. Endpoint : delta apparié par **sujet de test**,
à pli, seed, budget et beta identiques, moyenné sur la grille, IC bootstrap 20 000 sur
les sujets. Bilatéral, α = 0.05, Holm sur les 3 datasets.

**H_B (le vrai enjeu, et il est falsifiable).** L'écran hors ligne prédit le **signe** du
résultat end-to-end. Les trois signes sont gelés ici, avant le premier fit :

| dataset | signe prédit | prédit par |
|---|---|---|
| cho2017 | **+** (restored bat acc) | top-5 +0.0387 |
| lee2019_mi | **+** | top-5 +0.0143 |
| physionetmi | **−** (acc bat restored) | top-5 −0.0286 |

Le succès de H_B, c'est **3 signes sur 3**. 2/3 est un demi-échec et sera rapporté comme
tel — sous binomiale à p=0.5, 2/3 vaut p=0.5 et ne prouve rien. Aucune magnitude n'est
prédite : l'écran note un top-5, l'intervention note une AUC de receveur, les deux
échelles ne sont pas comparables et prétendre le contraire serait faux.

**Ce qui est explicitement abandonné.** L'hypothèse « quand les sujets se ressemblent,
l'accuracy cesse de discriminer et la croissance prend le relais » est **réfutée comme
mécanisme général** et n'est pas re-testée ici. Le gradient existait bien (monotone sur
4 datasets sur 5), mais le contrôle de symétrie — refaire la manipulation en fenêtrant
sur `params_probe` — montre que l'échange est **asymétrique dans le mauvais sens** sur
cho2017 (−0.110) et dreyer2023a (−0.203) : c'est surtout de la restriction d'étendue.
Il n'est réel que sur lee2019_mi (+0.140) et faible sur physionetmi (+0.038).
Voir `rule_dispersion.py`, section C.

## Puissance, mesurée et non supposée

MDE du delta apparié (puissance 0.80, bilatéral), calculé sur la dispersion **observée**
des deltas de `dalloc` :

| dataset | n | sd(delta) | SE | MDE |
|---|---|---|---|---|
| cho2017 | 52 | 0.0621 | 0.0086 | **+0.0241** |
| lee2019_mi | 54 | 0.0427 | 0.0058 | **+0.0163** |
| physionetmi | 109 | 0.0685 | 0.0066 | **+0.0184** |

Un nul sur lee2019_mi (MDE +0.016) sera lisible ; un nul sur cho2017 (MDE +0.024) le sera
moins et ne devra pas être écrit « restored ne fait pas mieux » — voir
[[underpowered-not-null]].

## Appariement — les clous à ne pas casser

Les cellules nouvelles doivent être **appariables une à une** avec celles de `dalloc/` :
mêmes plis (`donor_select.folds_of` importée, jamais réimplémentée), mêmes seeds 0/1/2,
mêmes budgets, mêmes betas 1/2/4/8, même checkout `eegrow_budget` épinglé à
`EEGROW_SHA=58c6bee`. Le beta=0 n'est **pas** relancé : il est déjà sur disque et ne
dépend d'aucune mesure. Sortie séparée dans `dalloc_restored/<dataset>` pour ne pas
mélanger les globs.

Le classement vient de `analysis/donor_select_prep.py`, jamais re-dérivé — les 7
`ranking_*.csv` ont été régénérés le 04/09 pour porter `restored_probe`, la moyenne des
15 réplicats de `restored_epoch`, relevée après `RestoreBestModel`.

## Grille et coût, mesurés sur la campagne finie

144 cellules par dataset : 4 plis × 3 budgets × 4 betas × 3 seeds.
Coût médian par cellule relevé sur les 1008 cellules `dalloc` terminées — cho2017
5.3 min, lee2019_mi 11.9 min, physionetmi 10.9 min — soit **≈ 68 GPU·h** au total.
