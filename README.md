# 🏠 Prédiction du prix des maisons (Kaggle *House Prices: Advanced Regression Techniques*)

[![tests](https://github.com/EddieZIDA/KaggleCompetition_home-price-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/EddieZIDA/KaggleCompetition_home-price-prediction/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)
![Optuna](https://img.shields.io/badge/Optuna-4.8-blue)
![Kaggle public LB](https://img.shields.io/badge/Kaggle%20public%20LB-0.11548%20RMSLE-20BEFF?logo=kaggle&logoColor=white)

Prédiction du prix de vente de 1 459 maisons à Ames (Iowa) à partir de 79 variables brutes, avec un
**pipeline scikit-learn sans fuite de données**, des modèles linéaires, à noyau et de gradient boosting
**optimisés avec Optuna**, et un **ensemble évalué honnêtement**.

Projet individuel, conçu et réalisé par **Wend Kouni Eddie Eliel ZIDA**.

| | |
|---|---|
| **Leaderboard public Kaggle** | **0.11548** (meilleur score v1 : 0.12277) |
| **Meilleur RMSLE en validation croisée** | **0.1071** ± 0.0080 (blend à poids optimisés, 5 plis) |
| Meilleur modèle seul | SVR (noyau RBF) : 0.1097 |
| Version précédente du dépôt | 0.1204 en CV ; sa dernière soumission d'ensemble était en échelle log et inutilisable, voir [les changements de la v2](#-ce-qui-a-changé-en-v2) |
| Amélioration | **−11 % d'erreur** |

![Comparaison des modèles](results/figures/model_comparison.png)

## 🚀 Démarrage rapide

```bash
git clone https://github.com/EddieZIDA/KaggleCompetition_home-price-prediction.git
cd KaggleCompetition_home-price-prediction
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS : source .venv/bin/activate
pip install -r requirements-dev.txt
```

Télécharger `train.csv` et `test.csv` depuis la
[page de la compétition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
dans `data/raw/`, puis :

```bash
python -m src.pipeline                              # CV de tous les modèles, ensemble, soumission (~5 min)
python -m src.pipeline --tune --trials 40 --tune-timeout 900   # relance d'abord Optuna (~50 min)
python -m src.pipeline --models ridge lasso svr --no-save      # expérience rapide
python -m src.predict data/raw/test.csv --out submissions/predictions.csv   # inférence avec l'ensemble sauvegardé
pytest                                              # 23 tests
```

`python -m src.pipeline` produit :

| Sortie | Contenu |
|---|---|
| `submissions/submission.csv` | soumission Kaggle validée (1 459 lignes, identifiants uniques, prix en dollars) |
| `models/final_ensemble.joblib` | ensemble entraîné, `predict_price(raw_dataframe)` |
| `models/params/*.json` | hyper-paramètres optimisés (versionnés) |
| `results/metrics.json`, `results/cv_results.csv` | scores, poids du blend, versions des bibliothèques |
| `results/oof_predictions.csv` | prédictions hors pli (out-of-fold) de chaque modèle |
| `results/figures/*.png` | figures utilisées dans ce README |

## 📈 Résultats

Validation croisée à 5 plis, RMSLE (= RMSE sur `log1p(SalePrice)`, la métrique de Kaggle). Tous les
scores sont hors pli : le prétraitement, les poids du blend et le méta-modèle du stacking ne voient
jamais le pli sur lequel ils sont évalués.

| Modèle | RMSLE CV | v1 (ancien README) |
|---|---|---|
| **Blend (poids optimisés)** | 0.1071 ± 0.0080 | n/a |
| **Stacking (méta-modèle linéaire positif)** | 0.1072 ± 0.0080 | n/a |
| SVR (noyau RBF) | 0.1097 ± 0.0098 | n/a |
| XGBoost | 0.1114 ± 0.0055 | 0.1246 |
| Lasso | 0.1117 ± 0.0069 | 0.1261 |
| ElasticNet | 0.1119 ± 0.0070 | n/a |
| CatBoost | 0.1122 ± 0.0057 | n/a |
| Gradient Boosting (Huber) | 0.1129 ± 0.0088 | n/a |
| LightGBM | 0.1130 ± 0.0077 | 0.1295 |
| Ridge | 0.1138 ± 0.0079 | 0.1297 |
| Random Forest (référence) | 0.1281 ± 0.0084 | 0.1412 |
| Régression linéaire (référence, non régularisée) | 0.2053 ± 0.1530 | divergente (1e11) |

Les scores v1 sont ceux annoncés dans l'ancien README ; ils provenaient d'un autre pipeline, avec
fuite de données, et ne sont donnés qu'à titre indicatif.

> **Remarque honnête sur le tuning.** Comme Optuna cherche sur d'autres plis que ceux utilisés pour
> les scores publiés, le tableau montre ce que le tuning apporte réellement : peu de chose. XGBoost
> optimisé obtient 0.1114 sur les plis d'évaluation contre 0.1104 avec des valeurs choisies à la main,
> deux scores dans le bruit d'un pli à l'autre (± 0.006). Les gros gains de la v2 viennent du travail
> sur les données (valeur manquante = absence, échelles ordinales, variables qualité × surface,
> correction de l'asymétrie : le Lasso passe de 0.1261 à 0.1117) et du mélange de modèles
> linéaires/à noyau avec des arbres boostés.

**Leaderboard public Kaggle**

| Soumission | RMSLE public |
|---|---|
| Meilleure soumission v1 | 0.12277 |
| Blend v2 | 0.12238 |
| Blend v2 + règle « grande maison en vente partielle » | **0.11548** |

Au départ, le gain obtenu en CV se voyait à peine sur le leaderboard. La cause était une seule maison
du jeu de test (Id 2550) : une maison de 5 095 pieds carrés à Edwards, vendue en `Partial`, jumelle
des 2 valeurs aberrantes d'entraînement (vendues 160 k$ et 185 k$). Ces jumelles étant retirées de
l'entraînement, aucun modèle ne pouvait l'apprendre et le blend l'extrapolait à 865 k$, une seule
erreur valant ≈ 0.007 de RMSLE. `EnsembleRegressor` attribue désormais aux maisons qui vérifient cette
règle (`GrLivArea > 4000 & Neighborhood == Edwards & SaleCondition == Partial`, soit exactement les
2 valeurs aberrantes d'entraînement et 1 maison de test, vérifié par un test) le prix log moyen de
leurs jumelles d'entraînement. Leçon : quand on retire des valeurs aberrantes de l'entraînement,
leurs équivalents dans le jeu de test demandent un traitement explicite.

**Poids du blend** (positifs, de somme 1, optimisés sur les prédictions hors pli) :

<p align="center"><img src="results/figures/blend_weights.png" width="560"></p>

Les modèles linéaires/à noyau et les arbres boostés font des erreurs assez différentes, c'est pourquoi
les combiner est utile.

![Prédictions hors pli](results/figures/oof_predictions.png)

<p align="center"><img src="results/figures/feature_importance.png" width="620"></p>

## 🔬 Méthodologie

**Données** (`src/data.py`)
- Les 2 maisons de plus de 4 000 pieds carrés vendues moins de 300 k$ (ventes partielles signalées
  par l'auteur du jeu de données) sont retirées, **dans le jeu d'entraînement uniquement** ; toutes
  les maisons de test sont prédites.
- Cible : `log1p(SalePrice)`, appliqué à un seul endroit ; `expm1` à un seul endroit.

**Variables** (`src/features.py`, toutes apprises sur le pli d'entraînement uniquement)
- *Valeur manquante = absence* : pour le garage, le sous-sol, la cheminée, la piscine, la clôture…
  une valeur manquante signifie « aucun » (l'EDA montre que les 81 `GarageType` manquants sont
  exactement les 81 maisons avec `GarageArea = 0`). Elles deviennent `"None"` / `0`, pas le mode.
- `LotFrontage` imputé par la médiane de son quartier ; `GarageYrBlt` des maisons sans garage (et la
  faute de frappe `2207` du jeu de test) remplacé par `YearBuilt`.
- Échelles de qualité (`Po` → `Ex`, finition du sous-sol, finition du garage, fonctionnalité…)
  encodées en **ordinal** ; `MSSubClass` et `MoSold` traitées comme des catégories.
- 15 variables construites : `TotalSF`, `QualTotalSF` (qualité × surface), `QualGrLivArea`,
  `TotalBathrooms`, `HouseAge`, `RemodAge`, `TotalPorchSF`, `OverallScore`, `IsNew`, `IsRemodeled`,
  `HasGarage`, `HasBsmt`, `Has2ndFlr`, `HasFireplace`, `HasPool`.
- Encodage one-hot appris sur l'entraînement (`handle_unknown="infrequent_if_exist"`, modalités rares
  regroupées).
- Pour les modèles linéaires et le SVR : `log1p` des variables asymétriques (asymétrie apprise sur
  l'entraînement) et `RobustScaler`.

**Modèles** (`src/models.py`) : Ridge, Lasso, ElasticNet, SVR (RBF), Gradient Boosting (perte de
Huber), XGBoost, LightGBM, CatBoost ; régression linéaire simple et random forest comme références.
Chaque modèle est un unique `Pipeline(prétraitement → estimateur)`, si bien que la validation croisée
réentraîne le prétraitement à chaque pli.

**Tuning** (`src/tuning.py`) : Optuna TPE avec une graine fixe, exécuté sur **d'autres découpages de
CV** (graine 2024) que ceux utilisés pour publier les scores (graine 42), afin de limiter le biais
optimiste qu'on obtient en optimisant et en évaluant sur les mêmes plis.

**Ensembles** (`src/ensemble.py`)
- *Blend* : poids positifs de somme 1, trouvés avec SLSQP sur les prédictions hors pli.
- *Stack* : méta-modèle linéaire à coefficients positifs sur les prédictions hors pli.
- Les deux sont évalués par une CV externe sur la matrice des prédictions hors pli ; le meilleur est
  réentraîné et utilisé pour la soumission.

## 🧰 Ce qui a changé en v2

Un audit de la v1 a montré que le pipeline ne pouvait pas produire de soumission valide. La v2 est
une réécriture :

| Problème de la v1 | Impact | v2 |
|---|---|---|
| `log1p` appliqué deux fois à `SalePrice` (notebook de prétraitement + notebook de modélisation) | tous les scores des notebooks sans valeur ; la soumission contenait des prix ≈ 11.7 $ | une seule transformation dans `src/data.py` ; `validate_submission` rejette les prix en échelle log |
| Jeu de test encodé avec son propre `OneHotEncoder` (`drop="first"`) | 99 % des maisons de test encodées avec un chauffage au sol, les toits `CompShg` comme des tuiles d'argile, 729 cheminées notées excellentes | un seul pipeline de prétraitement ajusté sur l'entraînement, réutilisé sur le test |
| Valeurs manquantes imputées par le mode | 690 maisons sans cheminée notées « Good » | gestion « manquant = absent » + échelles ordinales |
| Test imputé avec les statistiques du test ; valeurs aberrantes retirées du test dans `src/` | fuite de données ; soumission avec des lignes manquantes | toutes les statistiques apprises sur l'entraînement ; valeurs aberrantes retirées de l'entraînement uniquement |
| Les commandes `python -m src.*` plantaient (`KeyError`, variables incohérentes) ou sauvegardaient des modèles non entraînés | commandes du README inutilisables | une seule CLI testée : `python -m src.pipeline` |
| Optuna optimisait et évaluait sur les mêmes plis, sans graine | scores optimistes, non reproductibles | plis de tuning séparés, échantillonneur avec graine |
| LightGBM : `subsample` sans `subsample_freq`, `num_leaves` > 2^`max_depth` | paramètres optimisés sans effet | corrigé |
| Scores du README non produits par le code (par ex. 0.1184 en stacking) | résultats invérifiables | chiffres du README générés depuis `results/metrics.json` |
| Pas de tests, dépendances non figées, TensorFlow requis mais inutilisé | installation fragile | 23 tests pytest, CI, `requirements.txt` figé, TensorFlow retiré |

## 📁 Structure du projet

```
KaggleCompetition_home-price-prediction/
├── src/
│   ├── config.py          # chemins, graines, paramètres de CV
│   ├── data.py            # chargement, valeurs aberrantes, transformation de la cible
│   ├── features.py        # HouseFeatureEngineer, SkewCorrector, build_preprocessor()
│   ├── models.py          # catalogue de modèles (un Pipeline par modèle) + chargement des paramètres optimisés
│   ├── tuning.py          # espaces de recherche Optuna
│   ├── evaluation.py      # validation croisée hors pli
│   ├── ensemble.py        # blend / stack + EnsembleRegressor
│   ├── submission.py      # écriture et validation de la soumission
│   ├── visualization.py   # figures du README
│   ├── pipeline.py        # CLI : python -m src.pipeline
│   └── predict.py         # CLI : python -m src.predict
├── notebooks/
│   ├── 01_eda.ipynb             # analyse exploratoire
│   ├── 02_preprocessing.ipynb   # le pipeline de prétraitement, étape par étape
│   └── 03_modeling.ipynb        # CV, ensemble, analyse des erreurs, soumission
├── tests/                 # pytest (données synthétiques + tests d'intégration sur les fichiers Kaggle)
├── models/params/         # hyper-paramètres optimisés (versionnés)
├── results/               # métriques + figures (versionnées)
├── data/raw/              # CSV Kaggle (non versionnés)
├── .github/workflows/     # CI : ruff + pytest
├── requirements.txt / requirements-dev.txt
└── pyproject.toml
```

## 🎯 Pistes d'amélioration

- Validation croisée imbriquée, pour retirer complètement le biais du tuning du score publié.
- Target encoding de `Neighborhood` à l'intérieur des plis de CV ; gestion native des catégories dans CatBoost.
- Valeurs SHAP pour expliquer la prédiction de chaque maison.
- Moyenne sur plusieurs graines pour les modèles boostés.

## 👤 Auteur

Projet individuel : tout le travail (analyse, pipeline, modélisation, tests et documentation) a été
réalisé par **Wend Kouni Eddie Eliel ZIDA** ([GitHub](https://github.com/EddieZIDA) · [LinkedIn](https://linkedin.com/in/eddiezida)).

Données : Kaggle *House Prices: Advanced Regression Techniques* (Dean De Cock, jeu de données Ames
Housing). Projet pédagogique.
