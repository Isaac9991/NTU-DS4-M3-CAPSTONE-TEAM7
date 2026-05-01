import pandas as pd
import numpy as np

def preprocess_data(df, is_train=True):
    """
    Preprocess the dataset by extracting features, dropping duplicate or unneeded columns,
    and converting object types to categorical for LightGBM.
    """
    df = df.copy()
    
    # 1. Drop redundant/highly correlated or unneeded ID columns
    cols_to_drop = ['mid_storey', 'floor_area_sqft', 'lease_commence_date', 'Tranc_YearMonth']
    if 'id' in df.columns and is_train:
        cols_to_drop.append('id') # Keep ID in test set for submission
        
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # 2. Extract some basic features if needed
    # (Since Tranc_Year and Tranc_Month are already present, we don't strictly need to parse Tranc_YearMonth)
    
    # Optional logic: Handle highly sparse or text columns without meaning
    # e.g., 'address' might have too many unique values, so we just use 'block' and 'street_name'
    if 'address' in df.columns:
        df = df.drop(columns=['address'])
        
    # 3. Convert object columns to category
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category')
        
    # 4. Separate target if parsing training data
    target = None
    if is_train and 'resale_price' in df.columns:
        target = df['resale_price']
        df = df.drop(columns=['resale_price'])
        
    return df, target
