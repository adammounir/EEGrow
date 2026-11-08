# PREREG — Une taille qui s'adapte au sujet (lecture 4, axe 1)

08/11/2026. **GELÉ** dans le commit qui contient `adapt_C.py`, ses sbatch, son test et
`PREREG_growth_signature.md` (axe 2, même calcul), avant toute adaptation hors du dataset
de réglage. Aucun nouvel entraînement de modèle de base.

## 1. Question

Les lectures 2 et 3 donnaient à chaque sujet le même nombre de neurones. Résultat : la
croissance vaut LoRA, et fait un peu mieux que réentraîner la tête. La seule chose qu'une
croissance peut faire et qu'un adaptateur de taille fixe ne fait pas, c'est **choisir sa
taille pour chaque sujet**. On teste :

1. choisir la taille sur les données du sujet fait-il mieux que lui donner la taille
   choisie pour un autre sujet (même distribution de tailles) ?
2. à taille choisie de la même façon, la croissance bat-elle LoRA ?
3. choisir la taille fait-il mieux que la meilleure taille fixe ?

**Comment la taille est choisie.** Le papier TINY (Verbockhaven et al., TMLR 2024) ne
définit **aucune** règle pour le nombre de neurones : § 3 « a selection based on
statistical significance of singular values can also be performed », § 7 la renvoie aux
travaux futurs (« define a threshold to select neurons … based on statistical
significance »), et ses expériences ajoutent un nombre fixe par couche (Table 2). Lu dans le
PDF le 08/11. Une sonde (bnci2014_001, dataset de réglage) a aussi écarté un seuil par
permutation des labels : avec des labels mélangés, l'erreur du modèle augmente, donc le
spectre de référence dépasse le vrai à toutes les jonctions (taille 0 partout). On prend
donc la règle standard, la même pour les deux familles : **validation croisée interne à 2
plis sur les seuls essais de calibration**.

## 2. Données, modèles, découpages, tirages

Identiques à la lecture 3 (`PREREG_growth_adaptation_B.md` § 2-3), avec deux changements :
- **Modèles** : Shallow, EEGNeX, Conformer, MSVTNet. **CTNet exclu** : dans ses modèles
  partagés, la sortie du transformer est constante d'un essai à l'autre (écart-type 1e-6 à
  1e-8 pour une norme ~0,8, sonde exploratoire du 08/11, JOURNAL). Tout adaptateur de FFN y
  ajoute un biais commun et ne peut pas bouger l'AUC, si bien que la taille n'y a pas de sens.
  Les modèles retenus doivent aussi avoir passé G1 à la lecture 3 (les 4 l'ont passé).
- **k** ∈ {10, 50} essais par classe (10 = comme avant, 50 = assez de données pour que la
  taille compte). 6 adaptations par (sujet, k) comme la lecture 3, mêmes graines.

G0 (reconstruction du modèle de la campagne) : règle de la lecture 3, mêmes refus.

## 3. Bras

Pour un budget b ∈ {1, 2, 4, 8, 16} : `lora<b>` = LoRA de rang b, `grow<b>` = le nombre de
neurones par jonction dont le compte de paramètres est le plus proche de LoRA rang b (au
moins 1), aux jonctions et avec l'init et l'entraînement de la lecture 3 (AdamW, 100 pas, lot
complet, AdaBN avant et après). b = 0 est AdaBN. Le zéro-shot est calculé et rapporté.

**Taux d'apprentissage** : un par (modèle, bras), dans {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
1e-1}, réglé sur bnci2014_001 (9 sujets, k = 10) par le score moyen. La grille gagne 1e-1
parce que, à la lecture 3, EEGNeX grow2 choisissait le bord haut (3e-2).

**Meilleure taille fixe** `fix` : pour chaque (modèle, famille), le budget dont le meilleur
score de réglage est le plus haut (ex æquo : le plus petit). Choisie sur bnci2014_001 seul,
utilisée aux deux k.

**Taille choisie** (`grow_cv`, `lora_cv`), pour chaque tirage : les 2k essais de calibration
sont coupés en deux moitiés équilibrées par classe (`default_rng(graine du tirage)`). Pour
chaque b ∈ {0, 1, 2, 4, 8, 16}, on adapte sur une moitié et on note le score sur l'autre,
dans les deux sens. b* = le meilleur score moyen (ex æquo : le plus petit). Le score de
`grow_cv` est celui de `grow<b*>` réentraîné sur toute la calibration, testé sur la partie
de test. Rien de la partie de test ne sert au choix.

**Taille d'un autre** (`grow_perm`) : le sujet i reçoit le b* choisi pour un autre sujet π(i)
du même dataset, au même (k, sens, tirage). π est fixée avant tout calcul : ordre aléatoire
des sujets (`default_rng(0)`), chacun reçoit le suivant (permutation sans point fixe). Si
π(i) n'a pas ce k, le tirage ne compte pas pour i.

## 4. Critères

Différences appariées par sujet, moyennées sur ses tirages, puis sur les 4 modèles (un sujet
doit les avoir tous). Groupé sur les 11 datasets : moyenne des moyennes par dataset pondérées
par √n_d, IC95 bootstrap stratifié (10 000 tirages dans chaque dataset, `default_rng(0)`),
p unilatéral = part des tirages ≤ 0 (`adapt_B.sboot`, celui de la lecture 3).

**Principal**, Holm sur les 6 (3 critères × 2 k), α = 0,025 unilatéral :

| critère | contraste | question |
|---|---|---|
| **S1** | grow_cv − grow_perm | la taille du sujet compte-t-elle ? |
| **S2** | grow_cv − lora_cv | à taille choisie, croissance contre LoRA |
| **S3** | grow_cv − grow_fix | choisir contre la meilleure taille fixe |

Verdicts : SUPÉRIEUR si Holm rejette ; sinon NON INFÉRIEUR si borne basse > −0,02 ; sinon
INFÉRIEUR si borne haute < 0 ; sinon NON CONCLUANT (règle de la lecture 3, MDE rapportée).

**Secondaire** : S1-S3 par modèle, Holm sur les 24.

Lecture :
- S1 SUP aux deux k → la taille choisie pour le sujet vaut mieux qu'une taille d'un autre :
  « la taille s'adapte au sujet » est établi. S1 non SUP → la taille choisie ne porte rien de
  propre au sujet (ou la validation à 2k essais ne sait pas la trouver ; l'oracle le dit) ;
- S2 SUP → la croissance profite mieux du choix de taille que LoRA ; NI → à égalité ;
- S3 SUP → choisir bat la meilleure taille fixe ; sinon une taille fixe suffit.

## 5. Rapporté, sans valeur décisionnelle

lora_cv − lora_perm et lora_cv − lora_fix ; **oracle** (meilleur b par sujet d'après la
partie de test, donc optimiste) − fix pour les deux familles, qui borne ce qu'un choix de
taille peut gagner ; grow_fix − lora_fix ; répartition des tailles choisies par modèle, k et
famille ; scores moyens par bras ; taux et tailles fixes réglés.

## 6. Ce qui ne sera pas fait

Aucun critère, k, budget, bras, règle de choix ou taux changé après lecture. Pas de CTNet,
pas d'autre graine, pas de règle TINY de comptage (elle n'existe pas). Si le résultat est
positif, une réplication sur des sujets jamais vus (dreyer2023a, liu2024, modèles partagés
à entraîner) fera l'objet de son propre PREREG.

## 7. Limites connues avant lecture

- La validation interne à k = 10 se fait sur 5 essais par classe et par moitié : choix
  bruité. C'est la vraie question pratique (choisir avec ce qu'on a), et l'oracle borne ce
  qu'on perd.
- EEGNeX : un neurone coûte ~2 rangs, donc grow1 = grow2 = grow4 (1 neurone), grow8 = 2,
  grow16 = 4 ; budgets inégaux comme à la lecture 3.
- Réglage du lr et de `fix` sur un seul dataset, à k = 10, réutilisé à k = 50.
- Toutes les autres limites de la lecture 3 (§ 9) tiennent.
- Les 315 sujets ont déjà servi à la lecture 3 (autres bras). Aucun des contrastes ci-dessus
  n'y a été calculé, mais l'idée de l'axe vient de son résultat : c'est pourquoi une
  réplication sur des sujets neufs est prévue si le résultat est positif.

## 8. Coût et exécution (mesurés avant le gel)

RTX 2080 Ti, bnci2014_001 s1, un tirage, tous les bras + la validation interne : Shallow
11 s (k = 10) / 25 s (k = 50) ; EEGNeX 43 / 218 s ; Conformer 36 / 87 s ; MSVTNet 26 / 58 s.
Extrapolé aux 6 tirages, aux 315 sujets et aux montages plus grands : ~400 h-GPU (EEGNeX
~200), ~40 h sur 10 cartes ; réglage ~5 h-GPU. Cartes turing/ampere/hopper ≥ 10 Gio, TF32
coupé, margpu020 et margpu028 exclus, **au plus 10 tâches GPU à la fois**. Chaîne :
`tasks` → réglage (36 tâches) → `merge` (CPU) → adaptation → passe signature (axe 2) →
lecture des deux axes (CPU), en `afterok`. Une tâche reprise saute les sujets déjà écrits.
