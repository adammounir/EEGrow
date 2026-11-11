# EXPLO E1b-rank — supprimer les FFN abîme-t-il le classement des essais, ou seulement l'échelle des logits ?

**Statut : EXPLORATOIRE.** Note gelée avec `e1b_rank.py` avant toute lecture d'un score de ce script. Go d'Adam le 11/11/2026.

## Pourquoi

E1b-bias (`read/E1b_bias.md`) a montré deux choses sur Conformer :
- supprimer toutes les branches FFN au test (z_all) coûte de l'accuracy partout, ~2-3 points à 2 classes et ~4 points à 3-4 classes ;
- à 2 classes, l'AUC ne bouge presque pas (within −0,002). Le classement des essais y est donc intact.

Recentrer les logits par une constante ne récupère pas ce coût, et l'aggrave même à 2 classes. Ce n'était donc pas la bonne correction, et E1b-bias ne tranche pas entre deux explications :
- **échelle ou forme** : sans FFN, les logits changent d'échelle ou de forme, mais l'ordre des essais reste le même. Seul l'argmax se trompe ;
- **information** : sans FFN, le réseau ne sait plus séparer les classes.

Une mesure de rang aveugle à l'échelle sépare les deux.

## Mesure

Ce sont les mêmes fits et la même reconstruction que la copie figée `E1b_f0c8778` (import direct). Pour chaque groupe de test, on calcule les logits `full` et `z_all`, puis :
- `pw_*` : AUC par paires de classes, en float64. Pour chaque paire j < k, on prend les essais des classes j et k, avec le score l_k − l_j, et on calcule l'AUC. On fait ensuite la moyenne sur les paires (moyenne de Hand & Till 2001, appliquée aux écarts de logits au lieu des probabilités).
  - Elle ne change pas sous un décalage constant par classe, sous une mise à l'échelle positive de tous les logits, ni sous une constante par essai (la jauge du softmax).
  - À 2 classes, c'est l'AUC de l1 − l0.
  - Si une classe manque dans le groupe, elle vaut NaN, et le groupe est compté puis écarté du contraste.
- `ovr_*` : AUC un-contre-tous (macro) sur les log-odds float64, l_k − logsumexp(l_{j≠k}). Elle n'est pas invariante au décalage au-delà de 2 classes. Elle est seulement enregistrée.
- `acc_*` : accuracy argmax, comme dans E1b-bias.

## Portes (refus, pas avertissement)

- **G0** : la porte d'E1b (logits reconstruits = logits sauvés de la campagne).
- **G1** : `nat_full` et `nat_z` doivent égaler les scores de la copie figée d'E1b, et `acc_full` et `acc_z` ceux de la copie figée d'E1b-bias (`E1b_bias_e8e3468`), à 1e-6. Un seul échec refuse la lecture.
- **G2** : c'est la prémisse. Sur chaque groupe réel, pour `full` comme pour `z_all`, pw(2,5·l + c) doit égaler pw(l) à 1e-9, avec c un décalage aléatoire par classe. Un seul échec refuse la lecture.
- Contrôles synthétiques au selftest : invariance, égalité avec l'AUC à 2 classes, sortie constante = 0,5, classe manquante = NaN. Deux contrôles positifs : du bruit qui détruit l'information fait baisser pw, et un décalage qui fait chuter l'accuracy laisse pw intact.

## Décision, écrite avant lecture

On compare des contrastes appariés par sujet, avec les `sboot` d'E1b (moyenne des datasets pondérée par √n, sujets ré-échantillonnés dans chaque dataset).
- A = acc(z_all) − acc(full). Il est déjà connu par E1b-bias et rappelé seulement.
- K = pw(z_all) − pw(full).

**Primaire** : Conformer bd, datasets à 3-4 classes. Il y a un verdict par protocole :
- **RANG INTACT** si la borne basse de l'IC de K est > −0,010 ;
- **RANG PERDU** si la borne haute de l'IC de K est < −0,010 ;
- sinon **NON CONCLUANT**.

**Conclusion :**
- RANG INTACT sur au moins 2 des 3 protocoles : le coût des FFN de Conformer est un changement d'échelle ou de forme des logits, pas une perte de classement. L'axe « intérieur des transformers » est clos.
- RANG PERDU sur au moins 2 des 3 protocoles : les FFN de Conformer portent de l'information de classe sur les tâches multi-classes. On propose la suite à Adam, sans rien lancer avant.
- Sinon : non concluant, rapporté tel quel.

**Secondaires**, qui ne décident rien : tous les modèles et bras, les datasets à 2 classes (K y est l'AUC float64), l'AUC un-contre-tous, le pw médian de `full`, le détail par dataset.

## Limites connues avant lecture

- Peu de datasets multi-classes : 4 en within et en cross_subject, 2 en cross_session. La MDE sera rapportée.
- Le seuil −0,010 est en unités d'AUC et A en unités d'accuracy. Les deux échelles ne sont pas comparables terme à terme. On lit K seul contre son seuil, pas un rapport K/A.
- « Rang intact » veut dire que l'ordre par paires de classes est conservé. Ça ne dit pas quelle correction de sortie récupérerait l'accuracy. Le recentrage constant d'E1b-bias n'y suffisait pas.
- Le nombre de classes reste confondu avec l'identité du dataset.
