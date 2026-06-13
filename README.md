# Medical Insurance Cost Prediction - ANN

## Overview
A regression model using an Artificial Neural Network (ANN) to predict medical insurance charges based on personal and demographic attributes.

## Dataset
- **File:** `insurance_dataset.csv`
- **Rows:** 1338 (1337 after removing 1 duplicate)
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Target:** `charges`

## Data Preprocessing
- No missing values found.
- 1 duplicate row identified and dropped.
- Categorical encoding:
  - `sex`: male = 1, female = 0
  - `smoker`: yes = 1, no = 0
  - `region`: one-hot encoded into `region_northeast`, `region_northwest`, `region_southeast`, `region_southwest`
- Feature scaling: `StandardScaler` applied to both X and y (target was scaled separately using its own `StandardScaler`).
- Train-test split: 80/20.

## Model Architecture
```
Input(shape=(10,))
Dense(32, activation='relu')
Dense(16, activation='relu')
Dense(1, activation='linear')
```
- **Total Parameters:** 865
- **Optimizer:** Adam (learning_rate = 0.001)
- **Loss:** Mean Squared Error (MSE)
- **Metrics:** MAE, R² Score

## Training
- **Epochs:** Up to 100 (with Early Stopping)
- **Batch Size:** 32
- **Early Stopping:** Monitors `val_loss`, patience = 15, restores best weights

## Evaluation Results
| Metric | Value |
|---|---|
| Test Loss (MSE, scaled) | 0.1412 |
| Test MAE (scaled) | 0.2328 |
| Test R² Score | **0.8874** |

## Model Observations
- The model converges quickly, with R² improving from ~ -0.27 (epoch 1) to ~0.89 within 100 epochs.
- An R² of ~0.887 indicates the model explains about 88.7% of the variance in insurance charges — a strong fit for this small, mostly tabular dataset.
- Since both features and the target were standard-scaled, the loss/MAE values are in scaled units. To interpret in real dollar terms, use `scaler_y.inverse_transform()` on predictions.
- The relatively simple architecture (32 → 16 → 1) is sufficient given the small feature set (10 columns) and dataset size (~1337 rows); a deeper network would likely overfit.
- `smoker` status, `age`, and `bmi` are typically the most influential predictors of insurance cost in this type of dataset (consistent with domain knowledge, though feature importance was not explicitly computed in the notebook).

## Requirements
```
pandas
numpy
matplotlib
scikit-learn
tensorflow
```

## How to Run
1. Place `insurance_dataset.csv` in the same directory as the notebook.
2. Run all cells sequentially in Jupyter.
