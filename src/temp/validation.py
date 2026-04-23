import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import os

def generate_bins(target, num_bins=10):
    """
    Generate target bins for Stratified K-Fold.
    """
    bins = pd.qcut(target, q=num_bins, labels=False, duplicates='drop')
    return bins

def run_cv_pipeline(X_train, y_train, X_test, test_ids, n_splits=5, seed=42):
    """
    Runs a robust Stratified K-Fold CV using LightGBM.
    Returns the OOF predictions and Test predictions.
    """
    print(f"Starting {n_splits}-Fold Stratified CV with LightGBM...")
    
    # 1. Generate Stratified Bins for regression target
    bins = generate_bins(y_train, num_bins=10)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    cv_scores = []
    
    # Identify categorical columns
    cat_cols = X_train.select_dtypes(include=['category']).columns.tolist()
    
    # HistGradientBoostingRegressor Parameters
    model_params = {
        'loss': 'squared_error',
        'learning_rate': 0.05,
        'max_leaf_nodes': 31,
        'max_depth': None,
        'random_state': seed,
        'max_iter': 500,
        'early_stopping': True,
        'n_iter_no_change': 10
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, bins)):
        print(f"--- Fold {fold+1} ---")
        
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # HistGradientBoostingRegressor requires explicitly setting categorical features
        # It takes indices or a boolean mask
        cat_features_mask = [col in cat_cols for col in X_train.columns]
        model = HistGradientBoostingRegressor(**model_params, categorical_features=cat_features_mask)
        
        # Fit model
        model.fit(X_tr, y_tr)
        
        # Get predictions
        v_preds = model.predict(X_va)
        oof_preds[val_idx] = v_preds
        
        score = np.sqrt(mean_squared_error(y_va, v_preds))
        cv_scores.append(score)
        print(f"Fold {fold+1} RMSE: {score:.2f}")
        
        # Predict on Test set
        test_preds += model.predict(X_test) / n_splits
        
        
    print(f"\nMean CV RMSE: {np.mean(cv_scores):.2f} +/- {np.std(cv_scores):.2f}")
    
    # Create Output DataFrames
    oof_df = pd.DataFrame({'id': range(len(oof_preds)), 'predicted_price': oof_preds, 'actual_price': y_train})
    submission_df = pd.DataFrame({'id': test_ids, 'predicted_price': test_preds})
    
    return oof_df, submission_df
