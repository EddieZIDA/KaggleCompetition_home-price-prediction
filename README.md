# 🏆 Kaggle Competition : House Prices - Advanced Regression Techniques

Ce dépôt contient ma solution complète pour la célèbre compétition Kaggle **[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)**. 

L'objectif de cette compétition est de prédire le prix de vente final de maisons situées à Ames (Iowa) en utilisant 79 variables explicatives. Ce projet démontre une approche de bout en bout : de l'analyse exploratoire des données (EDA) jusqu'à l'optimisation avancée des modèles de Machine Learning.

## 📂 Structure du Projet

L'approche est modulaire et divisée en trois carnets (notebooks) :

* **`01_eda.ipynb`** : Analyse Exploratoire des Données (EDA). Étude de la distribution de la cible, détection des valeurs aberrantes (outliers) et analyse des corrélations.
* **`02_processing.ipynb` (Train & Test)** : Pipeline de préparation des données pour la soumission.
  * Imputation robuste des valeurs manquantes.
  * Transformation logarithmique (`log1p`) de la variable `SalePrice`.
  * **Feature Engineering** ciblé : Création de variables clés (`TotalSF`, `HouseAge`, `YearsSinceRemodel`) pour maximiser le signal prédictif.
  * Encodage (One-Hot) et alignement strict des dimensions entre les sets de Train et de Test.
* **`03_modeling.ipynb`** : Entraînement, évaluation et génération de la soumission.
  * Stratégie de validation : Cross-Validation (5-Fold).
  * Modèles évalués : Linear Regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM et un réseau de neurones (ANN).
  * Optimisation bayésienne des hyperparamètres avec **Optuna**.

## 🏅 Résultats et Soumission

La métrique d'évaluation officielle de la compétition est le **RMSLE** (Root Mean Squared Logarithmic Error).

* **Meilleur modèle individuel :** XGBoost
* **Score Public Kaggle (RMSLE) :** **0.12673**

*(Note : Des expérimentations avec des techniques d'ensembling comme le Stacking sont en cours pour améliorer davantage ce score).*

## 🛠️ Technologies & Outils

* **Langage :** Python 3
* **Data Science Stack :** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning :** Scikit-Learn, XGBoost, LightGBM, TensorFlow/Keras
* **Optimisation :** Optuna

## ⚙️ Reproduire la solution

1. Clonez ce dépôt :
   ```bash
   git clone [https://github.com/votre-username/nom-du-repo.git](https://github.com/votre-username/nom-du-repo.git)
