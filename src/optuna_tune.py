import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

from data import load_data
from feature_eng import preprocess_data
from feature_eng_location import add_location_features
from validation import generate_bins

# Load data once outside the objective for speed
print("Loading and preprocessing data for tuning...")
train_df = load_data('data/train.csv')
train_df, _ = add_location_features(train_df, None)
X_train, y_train = preprocess_data(train_df, is_train=True)

# Identify categorical columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
bins = generate_bins(y_train, num_bins=10)

def objective(trial):
    # Parameter search space
    params = {
        'loss_function': 'RMSE',
        'iterations': 1000, # Kept moderate for tuning speed
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'random_seed': 42,
        'early_stopping_rounds': 50,
        'verbose': 0,
        'task_type': 'CPU' # Set to 'GPU' if available
    }
    
    # 3-fold CV for faster tuning
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, bins)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, 
                  cat_features=cat_cols,
                  eval_set=(X_va, y_va),
                  use_best_model=True)
        
        v_preds = model.predict(X_va)
        score = np.sqrt(mean_squared_error(y_va, v_preds))
        cv_scores.append(score)
        
    mean_rmse = np.mean(cv_scores)
    return mean_rmse

if __name__ == '__main__':
    print(f"Starting Optuna Hyperparameter Optimization for CatBoost with {X_train.shape[1]} features...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, gc_after_trial=True)
    
    print("\nOptimization Finished!")
    print(f"Best trial value (RMSE): {study.best_value:.2f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
