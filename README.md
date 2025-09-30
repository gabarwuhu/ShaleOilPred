# ShaleOil-VA-RF
Code for shale oil production prediction and sweet spot evaluation using Variance-Adaptive Random Forest (VA-RF)

## Features
- Implements **Variance-Adaptive Random Forest (VA-RF)**: dynamically adjusts the split probability based on variance to balance tree strength and diversity, improving prediction stability under small-sample conditions.  
- Includes **Random Forest (RF)** and **Fixed-Probability RF (FPRF)** as baselines for comparison.  
- Uses **Bayesian Optimization** (`BayesSearchCV`) for automatic hyperparameter tuning.  
- Employs **Nested K-Fold Cross-Validation** (outer folds for evaluation, inner folds for tuning) to ensure robust generalization.  
- Provides **theoretical validation** with analysis of **tree strength** and **inter-tree correlation**.  
- Supports multiple evaluation metrics: **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.  
- Outputs results as **CSV files**, including per-fold errors and theoretical validation tables.  
- Includes a **sample dataset** (`Github示例数据.xlsx`) for quick testing.  

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


