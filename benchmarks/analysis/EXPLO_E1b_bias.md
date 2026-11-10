# EXPLO E1b-bias — le coût de supprimer les FFN est-il un décalage constant des logits ?

**Statut : EXPLORATOIRE.** Note gelée avec `e1b_bias.py` avant toute lecture d'un score de ce script. Go d'Adam le 10/11/2026.

## Pourquoi

Dans E1b, supprimer toutes les branches FFN au test (z_all) ne coûte presque rien. L'analyse exploratoire du 10/11 (`/scratch/amounir/eegrow_read/E1b_explo/datasize.log`) montre que ce coût n'est pas lié à la taille des données, mais au nombre de classes. Pour Conformer bd en within, il vaut environ +0,001 sur les 8 datasets à 2 classes et +0,040 sur les 4 datasets à 3-4 classes (alexmi, zhou2016, bnci2014_001, schirrmeister2017).

Le score d'E1b dépend du nombre de classes :
- à 2 classes, c'est la ROC-AUC, qui ne change pas si l'écart l1 − l0 se décale d'une constante ;
- à 3-4 classes, c'est l'accuracy, qu'un tel décalage peut faire basculer.

Le coût « multi-classes » peut donc venir d'un simple biais de sortie, et non d'une information de classe perdue.

## Mesure

Ce sont les mêmes fits et la même reconstruction que la copie figée `E1b_f0c8778` (import direct). Pour chaque groupe de test, on calcule les logits `full` et `z_all`. Puis :
- `acc_*` : accuracy argmax, sur tous les datasets, y compris à 2 classes ;
- `nat_*` : le score d'E1b ;
- `zrc` : logits z_all recentrés, `lg_z − moyenne_test(lg_z) + moyenne_test(lg_full)` par classe. Cette correction est constante et n'utilise pas les labels. Elle est transductive : elle ramène la sortie moyenne sur celle du modèle complet ;
- `bias_share` : part du changement de logits (centré sur les classes) qui est un décalage constant.

## Portes (refus, pas avertissement)

- **G0** : la porte d'E1b (logits reconstruits = logits sauvés de la campagne).
- **G1** : les scores `full` et `z_all` doivent égaler ceux de la copie figée d'E1b à 1e-6. Un seul échec refuse la lecture.
- **G2** : à 2 classes, l'AUC recentrée doit égaler l'AUC de z_all à 1e-9, calculée en float64 sur l1 − l0. C'est la prémisse de l'analyse. Un seul échec refuse la lecture.
  - Le premier selftest (job 921669) testait G2 sur le score natif. Il a refusé 4 groupes sur lee2019_mi s1, avec un |ΔAUC| ≤ 5e-3. Cause : le scorer passe par un softmax float32 qui sature, et le recentrage déplace les égalités (piège déjà connu, `checker-float32-softmax-ties`). La porte est donc passée en float64, et l'écart float32 est seulement compté (`nat32_moved`, `nat32_max_e6`).
  - Ce selftest n'imprime aucun score d'ablation. Seuls les |ΔAUC| des 4 groupes refusés ont été lus.

## Décision, écrite avant lecture

On compare des contrastes appariés par sujet, avec les `sboot` d'E1b (moyenne des datasets pondérée par √n, sujets ré-échantillonnés dans chaque dataset).
- A = acc(z_all) − acc(full).
- R = acc(z_all recentré) − acc(full).

**Primaire** : Conformer bd, datasets à 3-4 classes. Il y a un verdict par protocole :
- **BIAIS** si la borne basse de l'IC de R est > −0,010 (le recentrage récupère au moins ~75 % d'un coût de 0,04) ;
- **INFORMATION** si la borne haute de l'IC de R est < −0,010 ;
- sinon **NON CONCLUANT**.

**Conclusion :**
- BIAIS sur au moins 2 des 3 protocoles : les FFN de Conformer ne portent qu'un biais de sortie. L'axe « croissance dans les transformers » est clos.
- INFORMATION sur au moins 2 des 3 protocoles : les FFN portent de l'information de classe sur les tâches multi-classes. On propose la suite à Adam, sans rien lancer avant.
- Sinon : non concluant, rapporté tel quel.

**Secondaires**, qui ne décident rien : tous les modèles et bras, les datasets à 2 classes en accuracy, `bias_share`, le détail par dataset.

## Limites connues avant lecture

- Peu de datasets multi-classes : 4 en within et en cross_subject, 2 en cross_session (bnci2014_001, zhou2016). Les IC seront larges, et la MDE sera rapportée.
- Le recentrage utilise la sortie moyenne du modèle complet sur les mêmes essais. Ce n'est pas un modèle qu'on pourrait déployer. C'est un diagnostic de la forme du coût.
- Le nombre de classes reste confondu avec l'identité du dataset.
