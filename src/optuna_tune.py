import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

from data import load_data
from feature_eng import preprocess_data
from validation import generate_bins

def objective(trial):
    print(f"Starting Trial {trial.number}...")
    
    # Load data
    train_df = load_data('data/train.csv')
    X_train, y_train = preprocess_data(train_df, is_train=True)
    
    # Identify categorical columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Parameter search space
    params = {
        'loss_function': 'RMSE',
        'iterations': 1500, # Kept lower for tuning speed
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 10.0, log=True),
        'random_seed': 42,
        'early_stopping_rounds': 50,
        'verbose': 0
    }
    
    # Fast validation (3 folds instead of 5 for tuning)
    n_splits = 3
    bins = generate_bins(y_train, num_bins=10)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
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
    print(f"Trial {trial.number} finished with RMSE: {mean_rmse:.2f}")
    return mean_rmse

if __name__ == '__main__':
    print("Starting Optuna Hyperparameter Optimization for CatBoost...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, gc_after_trial=True)
    
    print("\nOptimization Finished!")
    print(f"Best trial value (RMSE): {study.best_value}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
