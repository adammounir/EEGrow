# PREREG — Contrôle d'init de Shallow (campagne A)

26/10/2026. **GELÉ** dans le commit qui contient aussi `GrowingShallowFBCSPNet(init=...)`,
`config/model/grow_shallow_bdinit.yaml`, les bras de contrôle de
`campaignA_make_cells.py` (`--arms tiny_bdinit,fix`) et la grille générée
(`benchmarks/campaignS_grid/`). Le gel a lieu avant toute cellule. Aucun contraste de la
campagne A n'a été lu.

## 1. Question

Dans la campagne A, le contraste Q1 de Shallow (tiny − bd) compare deux réseaux qui
diffèrent par **deux** choses.
- **La croissance**, ce qu'on veut mesurer.
- **L'initialisation.** `bd_shallow` est le ShallowFBCSPNet de braindecode, initialisé
  en `xavier_uniform_` (gain 1) avec des biais à 0 (braindecode 1.5.2,
  `shallow_fbcsp.py` l. 205-218). `grow_shallow` est la réécriture eegrow
  (`GrowingShallowFBCSPNet`), qui garde l'init par défaut de PyTorch : l'écart-type de
  conv_time est 2,6× plus grand et les biais sont non nuls (mesuré le 25/10).

L'architecture est identique : parité ≤ 1e-5 à poids copiés, et 46 084 paramètres de
part et d'autre. Les 4 autres modèles enveloppent les modules de braindecode et n'ont
donc pas ce problème.

Ce contrôle sépare les deux termes : **l'écart de Shallow tiny − bd survit-il quand le
réseau qui grandit part de l'init de braindecode ?**

## 2. Plan

### 2.1 Bras (Shallow seulement)

| bras | modèle | init | croissance | source |
|---|---|---|---|---|
| `bd` | `bd_shallow` | braindecode | non | campagne A (déjà dans la grille) |
| `tiny` | `grow_shallow` | PyTorch | tiny | campagne A (déjà dans la grille) |
| **`tiny_bdinit`** | `grow_shallow_bdinit` | braindecode | tiny | **ce contrôle** |
| **`fix`** | `fix_shallow` | PyTorch | non (40/40) | **ce contrôle** |

- **`tiny_bdinit`** tire chaque poids de départ de la loi de `bd_shallow`. Les bornes de
  Xavier sont celles de la géométrie cible 40/40, puisque la borne dépend de fan_out, et
  donc de la largeur. Le réseau de départ (8 filtres) est ainsi distribué exactement
  comme les 8 premiers filtres d'un réseau braindecode. Vérifié par
  `tests/test_shallow_init.py` : écart-type par tenseur à 5 % près et même borne sur 20
  graines, à 40 et à 8 filtres, biais à 0. Les neurones ajoutés suivent la règle tiny,
  comme dans `tiny`.
- **`fix`** est la classe de croissance construite directement à 40/40, sans croissance,
  sur la recette de `bd`. C'est le terme d'init sans la croissance.
- Les cellules sont **exactement** celles de la campagne A, au modèle près : le diff des
  lignes, une fois le nom du modèle retiré, est vide. Cela fait 2 405 cellules par bras,
  sur les 12 datasets, les 3 protocoles et les mêmes graines, sujets, recette, 200
  époques et score de la dernière époque. `tiny_bdinit` reprend les surcharges de
  `tiny`, et `fix` celles de `bd`.
- La grille gelée des 5 modèles, régénérée par le générateur modifié, est identique
  octet pour octet à celle qui tourne (`campaignA_cells.txt` et `campaignA_arrays/`).

### 2.2 Calcul

- 4 810 cellules, 407 tâches, 12 tableaux, environ 484 GPU-h en équivalent 2080 Ti
  (environ 5 % de la campagne A).
- Elles sont soumises avec `campaignA_b.sbatch` (même porte de cartes et mêmes
  exclusions que l'avenant 2 de la campagne A), depuis le checkout séparé
  `/scratch/amounir/eegrow_sinit` (ce commit). La vague en cours ne change pas de code.
- Les résultats vont sous `.../eegrow/campaignA/shallow__{tiny_bdinit,fix}/`.

## 3. Critères

Ce sont ceux de PREREG_campaign_A.md § 3, sans aucun changement : l'unité est le sujet,
les verdicts par case sont ceux du § 3.1, la lisibilité vient du bd, et Δ̄ groupé avec
bootstrap stratifié vient du § 3.2.

- **Endpoint principal, C1, la croissance à init égale** : A = `tiny_bdinit`, B = `bd`,
  un endpoint par protocole (3), avec Holm à α = 0,025 unilatéral sur ces 3. Verdicts
  SUPÉRIEURE, NON INFÉRIEURE, INFÉRIEURE ou NON CONCLUANTE, comme Q1.
- **Décomposition, rapportée sans décider** :
  - C2, l'init sous croissance : `tiny` − `tiny_bdinit` ;
  - C3, l'init sans croissance : `fix` − `bd` ;
  - par construction, C1 + C2 = Q1 de Shallow, sujet par sujet. C'est une identité,
    contrôlée par le lecteur pour s'assurer qu'il apparie bien les mêmes sujets.
- **Lecture de Q1 Shallow dans la campagne A** :
  - si C1 est SUPÉRIEURE, l'effet de Shallow est attribuable à la croissance ;
  - si Q1 est SUPÉRIEURE mais C1 ne l'est pas, Q1 de Shallow sera rapportée comme
    **confondue par l'init**, et le verdict publié sera C1 ;
  - Q2 (tiny − random) n'est pas concernée, puisque c'est la même classe et la même init.
- **Quand** : en même temps que la lecture unique de la campagne A (après TIDNet), avec
  le même `campaignA_check.py` sans problème. Le lecteur (`campaignA_read.py`, étendu
  aux paires de bras de ce contrôle) sera commité avant d'être lancé.

## 4. Ce qui ne sera pas fait

- Aucune init n'est essayée sur les 4 autres modèles : ils partagent déjà les modules
  de braindecode.
- On ne changera pas d'init après lecture, et on ne prendra pas la meilleure des deux
  comme résultat.
