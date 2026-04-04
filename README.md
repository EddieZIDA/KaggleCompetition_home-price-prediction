# 🏡 House Prices Prediction - Advanced Regression Techniques

Ce projet a pour objectif de prédire le prix de vente final de maisons situées à Ames (Iowa) en se basant sur un jeu de données contenant 79 variables explicatives (caractéristiques de la maison, de l'emplacement, de l'année de construction, etc.). 

L'approche adoptée couvre l'ensemble du cycle de vie d'un projet de Machine Learning : de l'analyse exploratoire initiale à l'optimisation avancée des modèles.

## 📂 Structure du Projet

Le projet est divisé en trois carnets (notebooks) principaux pour une meilleure lisibilité :

* **`01_eda.ipynb`** : Analyse Exploratoire des Données (EDA). Visualisation de la distribution de la variable cible (`SalePrice`), identification des valeurs manquantes et détection des valeurs aberrantes (outliers).
* **`02_processing.ipynb` (Train & Test)** : Nettoyage et préparation des données. 
  * Imputation des valeurs manquantes.
  * Transformation logarithmique (`log1p`) de la variable cible pour corriger l'asymétrie.
  * Encodage des variables catégorielles (One-Hot Encoding).
  * **Feature Engineering** : Création de variables à fort pouvoir prédictif (`TotalSF`, `HouseAge`, `YearsSinceRemodel`, etc.).
* **`03_modeling.ipynb`** : Entraînement et évaluation des modèles.
  * Évaluation robuste via Cross-Validation (5-Fold) sur la métrique **RMSLE**.
  * Modèles testés : Linear Regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM.
  * Optimisation des hyperparamètres du meilleur modèle à l'aide d'**Optuna**.

## 🛠️ Technologies Utilisées

* **Langage :** Python 3
* **Manipulation de données :** Pandas, NumPy
* **Visualisation :** Matplotlib, Seaborn
* **Machine Learning :** Scikit-Learn, XGBoost, LightGBM
* **Optimisation :** Optuna

## 🚀 Résultats

Le modèle **XGBoost** s'est démarqué comme étant le plus performant lors de la validation croisée. Après optimisation, le score RMSLE public obtenu démontre une forte capacité de généralisation du modèle sur des données invisibles.

## ⚙️ Comment utiliser ce projet

1. Clonez ce dépôt :
   ```bash
   git clone [https://github.com/votre-username/nom-du-repo.git](https://github.com/votre-username/nom-du-repo.git)
