# PREREG — Adapter un modèle partagé à un nouveau sujet en lui ajoutant des neurones (lecture 2)

07/11/2026. **GELÉ** dans le commit qui contient aussi `adapt_A.py` et son test, avant
toute adaptation sur lee2019_mi. Aucun nouvel entraînement de modèle de base : les modèles
partagés sont les fits cross-subject de la campagne A, déjà sauvés.

## 1. Question et motivation

Ce qu'on sait déjà (lu le 07/11) :
- lecture 1 (`read/embed_A.md`) : des neurones ajoutés en cours d'entraînement portent le
  sujet (top-1 0,324 contre 0,141 pour bd entre deux sessions disjointes), quelle que soit
  leur direction d'insertion (random 0,315) ;
- trajectoire (`read/embed_growth.md`, exploratoire) : cette empreinte est nulle à
  l'insertion et elle est écrite par l'entraînement qui suit. Les neurones ajoutés en
  prennent environ deux fois plus que les neurones initiaux.

Si des neurones qui partent de zéro absorbent ce qui est propre au sujet, ajouter des
neurones à un modèle partagé devrait être une bonne façon de l'adapter à un nouveau sujet.
LoRA est la méthode de référence pour ce rôle et part aussi de zéro (B = 0).

**Question** : pour adapter à un nouveau sujet un Shallow entraîné sur les autres sujets,
avec quelques essais de calibration, ajouter des neurones fait-il au moins aussi bien que
LoRA, et mieux que réentraîner la tête, à nombre de paramètres égal ?

Ce que les deux méthodes peuvent exprimer n'est pas la même chose, et c'est ce qui rend la
comparaison informative. Dans Shallow, `conv_time` (filtres temporels) alimente
`conv_spat` (filtres spatiaux) sans non-linéarité entre les deux :
- **LoRA sur conv_spat** ajoute ΔW = B·A de rang r : il **re-mélange** les 40 filtres
  temporels existants avec de nouveaux poids spatiaux ; il ne peut pas créer de nouvelle
  bande de fréquence ;
- **ajouter m neurones** ajoute m **nouveaux filtres temporels**, chacun avec ses poids
  spatiaux : il peut capter une bande que le modèle partagé n'utilisait pas.

## 2. Données

- **Modèles partagés** : `shallow__bd`, `cross_subject`, lee2019_mi, graine 0, poids
  finaux (`__fits__final`). Le fit du sujet i est entraîné sur les 53 autres et n'a jamais
  vu i. Architecture : `braindecode.models.ShallowFBCSPNet(n_chans=62, n_outputs=2,
  n_times=1000)` avec les arguments par défaut (`config/model/bd_shallow.yaml`,
  `module_kwargs: {}`), pipeline = conversion float32 puis réseau (`pipelines.py:347`).
- **Données du sujet i** : construites par `check_cache.build_campaign("lee2019_mi")`, lues
  dans le cache que lisent les cellules lee2019 (`cache_of`, soit
  `/scratch/amounir/moabb_cache`), en lecture seule ; un cache manquant refuse.
  2 sessions de 100 essais, 50 par classe (manifeste `manifest_A/lee2019_mi.json`).
- **Calibration et test** : on calibre sur une session et on teste sur **toute l'autre
  session** (100 essais), puis l'inverse. Les essais de calibration sont k par classe,
  tirés sans remise par `default_rng(1000·i + 10·d + r)` (i sujet, d direction, r tirage).
  3 tirages par direction. **Score d'un sujet** = moyenne des 2 directions × 3 tirages.
- **Score** : celui de la campagne A (`campaignA_check.score_of`) : ROC-AUC à 2 classes
  (lee2019_mi), accuracy sinon (bnci2014_001), sur les logits du test. La marge de −0,02
  garde donc le sens qu'elle avait dans la campagne.
- **k** (essais par classe) : 5, 10, 25, 50. **k = 10 est le critère principal** (20
  essais, l'ordre de grandeur d'une calibration de BCI).

## 3. Garde-fou G0 : on reconstruit bien le modèle de la campagne

Pour chaque sujet, le modèle reconstruit doit redonner les logits de test sauvés par la
campagne (`__fits__logits/<uid>__g{0,1}.npz`), sur les mêmes essais (`test_idx`, même
`y`) : argmax identique sur **tous** les essais et |Δ logit| ≤ 1e-3. Un sujet qui échoue
est refusé et compté ; si plus de 2 sujets échouent, le script refuse de produire les
critères (le chargement des données ou du modèle n'est pas celui de la campagne).

Vérifié avant le gel (07/11, pour valider le chargement, sans aucune adaptation lue) :
G0 passe sur lee2019_mi sujet 3 (max |Δ| 2,1e-5 sur les 2 sessions) et sur bnci2014_001
sujet 1 (1,5e-5). Coût mesuré sur 4 cœurs : ~20 s par adaptation à k = 10 (p = 8),
~80 s pour le fine-tuning complet à k = 50.

## 4. Bras d'adaptation

Tous partent du même modèle partagé, avec la même procédure :
1. **AdaBN** : les statistiques de la BatchNorm sont recalculées sur les essais de
   calibration (passe avant en mode train, sans gradient, `momentum=None`), puis figées ;
2. on entraîne les seuls paramètres du bras ; tout le reste est gelé, la BatchNorm reste
   en mode eval, le dropout en mode train ;
3. à la fin, les statistiques de la BatchNorm sont recalculées une fois de plus sur la
   calibration (une adaptation qui change l'entrée de la BatchNorm la rend sinon caduque ;
   pour la tête seule, c'est sans effet).

| bras | paramètres entraînés | nombre (p = 2 / p = 8) |
|---|---|---|
| zéro-shot | aucun, sans AdaBN | 0 |
| AdaBN | aucun | 0 |
| tête | `final_layer.conv_classifier` | 4 882 |
| **LoRA r = p** | A (r × 2 480), B (40 × r) sur `conv_spat` vu comme 40 × (40·62) ; B = 0, A Kaiming-uniforme (a = √5, comme peft) ; échelle α/r = 1 | 5 040 / 20 160 |
| **croissance m = p** | m nouveaux filtres temporels (25 poids + 1 biais, init PyTorch par défaut d'une Conv2d) et leurs poids spatiaux (40 × 62 par neurone), **initialisés à 0** | 5 012 / 20 048 |
| fine-tuning complet | tous | 105 202 (bd entier) |

- À p = 2, la tête, LoRA et la croissance ont le même nombre de paramètres à 3 % près.
  **p = 2 est le critère principal** ; p = 8 est rapporté.
- La croissance est implémentée par une branche parallèle : x → conv_time_new → conv_spat_new,
  ajoutée à la sortie de `conv_time_spat` avant la BatchNorm. Comme Shallow n'a pas de
  non-linéarité entre les deux convolutions, c'est exactement un `conv_time` de largeur
  40 + m suivi d'un `conv_spat` élargi. Poids spatiaux à 0 → la fonction de départ est
  exactement celle du modèle partagé, comme pour LoRA.
- Pas de règle TINY : la lecture 1 (P2) et la trajectoire disent que la direction compte
  peu, et ce choix aligne la croissance sur LoRA (départ nul, seul le gradient façonne).

**Entraînement** (tous les bras) : AdamW, décroissance 0, lot complet (≤ 100 essais),
100 pas, perte d'entropie croisée. La graine de l'init et du dropout est celle du tirage.

**Taux d'apprentissage** : un par bras, choisi dans {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} sur un
**autre dataset**, bnci2014_001 (9 sujets, fits cross-subject `shallow__bd` de la campagne
A, cache `moabb_cache_A`, même protocole, k = 10, p = 2 et p = 8 séparément), par
le score moyen de test. Le choix est écrit dans `tune.json` puis lu tel quel pour
lee2019_mi ; lee2019_mi n'est jamais utilisé pour choisir.

## 5. Garde-fou G1 : l'adaptation marche-t-elle du tout ?

Contrôle positif : à k = 50, **fine-tuning complet − zéro-shot** doit avoir sa borne basse
> 0. Sinon la recette d'adaptation est cassée, et E1 à E3 sont rapportés sans verdict.

## 6. Critères (lee2019_mi, n = 54, k = 10, p = 2)

Différences appariées par sujet, IC95 bootstrap percentile sur les sujets (10 000 tirages,
`default_rng(0)`), p unilatéral de supériorité = fraction des moyennes bootstrap ≤ 0,
Holm sur les 3 critères. La marge de non-infériorité est celle de la campagne A (−0,02).

| critère | contraste | question |
|---|---|---|
| **E1** | croissance − LoRA | ajouter des neurones vaut-il LoRA ? |
| **E2** | croissance − tête | ajouter de la capacité bat-il le réentraînement de la tête ? |
| **E3** | croissance − AdaBN | apprendre des paramètres apporte-t-il plus que recalibrer la BatchNorm ? |

Verdict de chacun : SUPÉRIEUR si Holm rejette sur la supériorité ; sinon NON INFÉRIEUR si
la borne basse > −0,02 ; sinon INFÉRIEUR si la borne haute < 0 ; sinon NON CONCLUANT.

Lecture :
- E1 SUPÉRIEUR ou NI, et E2 SUPÉRIEUR → la croissance est une méthode d'adaptation
  crédible face à LoRA (le papier de la lecture 2 a une base) ;
- E1 INFÉRIEUR → LoRA fait mieux à budget égal : la croissance n'apporte rien comme
  adaptateur sur Shallow ;
- E3 non SUPÉRIEUR → avec 20 essais, apprendre des paramètres n'apporte rien au-delà de la
  BatchNorm : la question est mal posée à ce k, regarder k = 25 et 50 (rapportés).

## 7. Rapporté, sans valeur décisionnelle

- La grille complète k ∈ {5, 10, 25, 50} × p ∈ {2, 8}, tous les bras, avec les MDE.
- Fine-tuning complet, zéro-shot, AdaBN à chaque k.
- Le gain de chaque bras sur zéro-shot en fonction de l'accuracy zéro-shot du sujet
  (descriptif, avec la mise en garde de la régression vers la moyenne).
- Les taux d'apprentissage choisis et la courbe de réglage de bnci2014_001.

## 8. Ce qui ne sera pas fait

- Aucun critère, k, p, bras ou taux d'apprentissage ne sera changé après lecture.
- Pas de règle TINY, pas de LoRA sur d'autres couches, pas de combinaison tête + croissance
  dans cette lecture.
- Pas d'autre modèle que Shallow, pas d'autre dataset que lee2019_mi pour les critères.

## 9. Limites connues avant lecture

- Un seul modèle partagé par sujet (graine 0) : la variance due au modèle partagé n'est
  pas mesurée.
- 100 pas à lot complet est un choix ; le réglage du taux d'apprentissage sur bnci2014_001
  (22 électrodes, 4 classes) peut mal se transposer à lee2019_mi (62 électrodes, 2
  classes). C'est le prix d'un réglage qui ne touche pas aux données de test.
- Sujet et session sont confondus avec le montage du casque (lecture 1, § 9).

## 10. Avenant 1 (08/11/2026) — G0 tolérant au TF32, avant tout critère

**Ce qui s'est passé.** La lecture (job 912307) a refusé de produire les critères : G0
échoue sur les sujets 46 à 54 (max |Δ logit| de 2,5e-3 à 7,1e-3), et passe sur 1 à 45
(~1e-5). Aucun critère, aucun score d'adaptation n'a été produit ni regardé : le rapport
ne contient que la liste des échecs de G0.

**Cause, établie indépendamment des résultats d'adaptation.** Selon les logs de la
campagne A, les fits partagés des sujets 46 à 54 ont été entraînés et ont sauvé leurs
logits sur des A40. Sur ampere/hopper, cuDNN calcule les convolutions en TF32 par défaut ;
le modèle reconstruit ici sur CPU, en float32 complet, en diffère de ~5e-3. Le même
mécanisme a été trouvé et mesuré pendant la préparation de la lecture 3, avant ce refus,
et la règle de G0 de la lecture 3 a été gelée pour cette raison en 8f4adbf. La
vérification d'avant gel de ce PREREG (§ 3) ne portait que sur deux sujets dont les fits
venaient de cartes sans TF32 : c'est une erreur de ma part.

**Changement.** G0 est recalculé pour les 54 sujets avec la règle de la lecture 3
(`adapt_B.g0`, PREREG_growth_adaptation_B.md § 3, inchangée) : par session,
|Δ logit| ≤ 1e-2 · max(1, max |logit|), argmax identique sauf sur les essais dont l'écart
entre les deux premières classes est ≤ 2 · max |Δ|, et |Δ score| ≤ 5e-3. Il est calculé
sur CPU (`adapt_A.py g0av1`, `out/g0_av1/<sujet>.json`), sur les mêmes données et le même
modèle reconstruit. La lecture refuse si ce G0 manque pour un sujet ; un sujet qui échoue
est exclu ; plus de 2 échecs → refus, comme avant. Le rapport donne aussi les échecs sous
la règle d'origine.

**Inchangé.** Les scores d'adaptation ne sont pas recalculés (`out/runs/` et `tune.json`
sont copiés tels quels) ; bras, taux, k, critères E1-E3, Holm, marge NI, G1 et contrastes
rapportés restent ceux des §§ 4-7. Le G0 du réglage sur bnci2014_001 avait passé sous la
règle d'origine.
