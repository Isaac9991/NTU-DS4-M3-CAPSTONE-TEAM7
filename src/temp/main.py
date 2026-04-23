import os
import pandas as pd
from pathlib import Path
from data import load_data
from src.temp.feature_eng import preprocess_data
from src.temp.validation import run_cv_pipeline

def main():
    print("Loading data...")
    train_df = load_data('data/train.csv')
    test_df = load_data('data/test.csv')
    
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    print("Preprocessing data...")
    # Get IDs for test set
    if 'id' in test_df.columns:
        test_ids = test_df['id'].values
    else:
        test_ids = range(len(test_df))
        
    X_train, y_train = preprocess_data(train_df, is_train=True)
    X_test, _ = preprocess_data(test_df, is_train=False)
    
    # Align train and test columns
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    X_test = X_test[X_train.columns]
    
    # Run validation pipeline
    oof_df, submission_df = run_cv_pipeline(X_train, y_train, X_test, test_ids, n_splits=5, seed=42)
    
    print("\nExporting predictions...")
    output_dir = Path(__file__).parent.parent / 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    oof_df.to_csv(output_dir / 'oof_predictions.csv', index=False)
    submission_df.to_csv(output_dir / 'submission.csv', index=False)
    print(f"Predictions saved to {output_dir}")
    
if __name__ == '__main__':
    main()
