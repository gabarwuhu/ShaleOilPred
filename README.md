# ShaleOil-VA-RF
Code for shale oil production prediction and sweet spot evaluation using Variance-Adaptive Random Forest (VA-RF)
## Features

Implements Variance-Adaptive Random Forest (VA-RF): dynamically balances node split randomness with variance information to improve model stability and accuracy.

Includes Random Forest (RF) and Fixed-Probability RF (FPRF) as baselines for comparison.

Uses Bayesian Optimization (BayesSearchCV) for automatic hyperparameter tuning.

Employs Nested K-Fold Cross-Validation (outer folds for evaluation, inner folds for tuning) to ensure robust performance.

Provides theoretical validation with tree strength and inter-tree correlation analysis.

Supports multiple evaluation metrics: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

Includes a sample dataset (Github示例数据.xlsx) with well names, geological/engineering features, and production targets for quick testing.

##Dataset
The repository provides a simple example dataset:

File: Github-data.xlsx

##Format:

First column → well names (not used in training).

Middle columns → geological/engineering features.

Last column → target variable.

## Installation
Ensure you have Python 3.12 installed. You can install the required dependencies using:

```bash
pip install numpy pandas scikit-learn scikit-optimize numba
```

## Usage
Run the main script to perform shale oil production prediction:

```bash
VA-RF.py
```

Modify hyperparameters or adjust K-Fold settings in the script as needed.

## Dependencies
The project requires the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-optimize`
- `numba`


