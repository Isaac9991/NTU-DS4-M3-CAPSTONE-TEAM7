import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

def get_kfold_target_encoding(df, col, target_col, n_splits=5):
    """Computes target encoding using K-Fold to prevent target leakage."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded_col = np.zeros(len(df))
    global_mean = df[target_col].mean()
    
    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        mean_target = train_df.groupby(col)[target_col].mean()
        encoded_col[val_idx] = val_df[col].map(mean_target).fillna(global_mean).values
        
    return encoded_col

def add_location_features(train_df, test_df=None):
    """
    Performs comprehensive location feature engineering safely without data leakage.
    If test_df is provided, it handles both to ensure consistent encodings.
    """
    print("Starting location feature engineering...")
    
    # Mark train and test
    train_df = train_df.copy()
    train_df['_is_test'] = False
    
    if test_df is not None:
        test_df = test_df.copy()
        test_df['_is_test'] = True
        df = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)
    else:
        df = train_df
        
    new_features = []
    
    # Convert month to datetime for time-based aggregations
    if 'Tranc_YearMonth' in df.columns:
        df['date'] = pd.to_datetime(df['Tranc_YearMonth'])
    else:
        raise ValueError("Column 'Tranc_YearMonth' is required for time aggregations.")
        
    # ---------------------------------------------------------
    # 1. Town-level encodings
    # ---------------------------------------------------------
    print("Processing Town encodings...")
    # Frequency encoding (based on full dataset is generally fine, but train-only is stricter)
    df['town_freq_enc'] = df['town'].map(df[~df['_is_test']]['town'].value_counts(normalize=True))
    new_features.append('town_freq_enc')
    
    # Target encoding using 5-fold CV for train, and global mean for test
    if 'resale_price' in df.columns:
        # Split back to compute encoding
        train_idx = df[~df['_is_test']].index
        test_idx = df[df['_is_test']].index
        
        # OOF encoding for train
        df.loc[train_idx, 'town_target_enc_cv'] = get_kfold_target_encoding(df.loc[train_idx], 'town', 'resale_price', n_splits=5)
        
        # Global mean for test
        if len(test_idx) > 0:
            global_means = df.loc[train_idx].groupby('town')['resale_price'].mean()
            overall_mean = df.loc[train_idx, 'resale_price'].mean()
            df.loc[test_idx, 'town_target_enc_cv'] = df.loc[test_idx, 'town'].map(global_means).fillna(overall_mean)
            
        new_features.append('town_target_enc_cv')
        
    # One-hot encoding of town
    town_dummies = pd.get_dummies(df['town'], prefix='town', drop_first=False)
    df = pd.concat([df, town_dummies], axis=1)
    new_features.extend(town_dummies.columns.tolist())
    
    # ---------------------------------------------------------
    # 2. Street-level encodings
    # ---------------------------------------------------------
    print("Processing Street encodings...")
    df['street_freq_enc'] = df['street_name'].map(df[~df['_is_test']]['street_name'].value_counts(normalize=True))
    new_features.append('street_freq_enc')
    
    if 'resale_price' in df.columns:
        train_idx = df[~df['_is_test']].index
        test_idx = df[df['_is_test']].index
        
        # OOF encoding for train
        df.loc[train_idx, 'street_target_enc_cv'] = get_kfold_target_encoding(df.loc[train_idx], 'street_name', 'resale_price', n_splits=5)
        
        # Global mean for test
        if len(test_idx) > 0:
            global_means = df.loc[train_idx].groupby('street_name')['resale_price'].mean()
            overall_mean = df.loc[train_idx, 'resale_price'].mean()
            df.loc[test_idx, 'street_target_enc_cv'] = df.loc[test_idx, 'street_name'].map(global_means).fillna(overall_mean)
            
        new_features.append('street_target_enc_cv')
        
    # ---------------------------------------------------------
    # 3. Block-level features
    # ---------------------------------------------------------
    print("Processing Block features...")
    # Extract numeric block number
    df['block_num'] = df['block'].str.extract(r'(\d+)').astype(float)
    new_features.append('block_num')
    
    # KMeans clusters (Fit on train, predict on both)
    block_nums_filled = df[['block_num']].fillna(0)
    kmeans5 = KMeans(n_clusters=5, random_state=42, n_init=10).fit(block_nums_filled.loc[~df['_is_test']])
    kmeans10 = KMeans(n_clusters=10, random_state=42, n_init=10).fit(block_nums_filled.loc[~df['_is_test']])
    
    df['block_num_cluster_5'] = kmeans5.predict(block_nums_filled)
    df['block_num_cluster_10'] = kmeans10.predict(block_nums_filled)
    new_features.extend(['block_num_cluster_5', 'block_num_cluster_10'])
    
    # Count of transactions per block in the past 12 months (Use only train data for counts to avoid future leak)
    # Actually, counting transactions in the test set is only valid if those transactions happened in the past.
    # But for a Kaggle competition, we only have train transactions as "past".
    # So we group by block/date using ONLY train transactions.
    train_only = df[~df['_is_test']]
    monthly_block_counts = train_only.groupby(['block', 'date']).size().reset_index(name='count')
    monthly_block_counts.set_index('date', inplace=True)
    
    def compute_rolling_past_12m(group):
        g = group.resample('MS').sum()
        return g['count'].shift(1).rolling(12, min_periods=1).sum()
        
    past_12m_counts = monthly_block_counts.groupby('block').apply(compute_rolling_past_12m).reset_index(level=0)
    past_12m_counts.rename(columns={'count': 'block_past_12m_tx_count'}, inplace=True)
    past_12m_counts.reset_index(inplace=True)
    
    # Merge back to the FULL dataframe
    df = df.merge(past_12m_counts, on=['block', 'date'], how='left')
    df['block_past_12m_tx_count'] = df['block_past_12m_tx_count'].fillna(0)
    new_features.append('block_past_12m_tx_count')
    
    # ---------------------------------------------------------
    # 4. Town-level market activity
    # ---------------------------------------------------------
    print("Processing Town-level market activity...")
    # Use only train data to compute historical town stats
    monthly_town_stats = train_only.groupby(['town', 'date']).agg(
        tx_count=('resale_price', 'size'),
        sum_price=('resale_price', 'sum')
    ).reset_index()
    monthly_town_stats.set_index('date', inplace=True)
    
    def compute_town_rolling_stats(group):
        g_counts = group['tx_count'].resample('MS').sum()
        g_sum_price = group['sum_price'].resample('MS').sum()
        
        # Current year (past 1-12 months)
        past_1_12m_counts = g_counts.shift(1).rolling(12, min_periods=1).sum()
        past_1_12m_sum_price = g_sum_price.shift(1).rolling(12, min_periods=1).sum()
        avg_price_current_year = past_1_12m_sum_price / past_1_12m_counts
        
        # Last year (past 13-24 months)
        past_13_24m_counts = g_counts.shift(13).rolling(12, min_periods=1).sum()
        past_13_24m_sum_price = g_sum_price.shift(13).rolling(12, min_periods=1).sum()
        avg_price_last_year = past_13_24m_sum_price / past_13_24m_counts
        
        res = pd.DataFrame({
            'transactions_in_town_last_year': past_1_12m_counts,
            'avg_price_town_current_year': avg_price_current_year,
            'avg_price_town_last_year': avg_price_last_year
        })
        res['price_trend_town'] = res['avg_price_town_current_year'] - res['avg_price_town_last_year']
        return res
        
    town_rolling_stats = monthly_town_stats.groupby('town').apply(compute_town_rolling_stats).reset_index(level=0)
    town_rolling_stats.reset_index(inplace=True)
    
    df = df.merge(town_rolling_stats, on=['town', 'date'], how='left')
    new_features.extend(['transactions_in_town_last_year', 'avg_price_town_current_year', 
                         'avg_price_town_last_year', 'price_trend_town'])
    
    # ---------------------------------------------------------
    # 5. Interaction features
    # ---------------------------------------------------------
    print("Processing Interaction features...")
    df['town_year'] = df['town'] + '_' + df['date'].dt.year.astype(str)
    df['town_flat_type'] = df['town'] + '_' + df['flat_type'].astype(str)
    
    if 'floor_area_sqm' in df.columns:
        df['block_num_x_area'] = df['block_num'] * df['floor_area_sqm']
    else:
        df['block_num_x_area'] = np.nan
        
    new_features.extend(['town_year', 'town_flat_type', 'block_num_x_area'])
    
    # Clean up temp columns
    df = df.drop(columns=['date', '_is_test'])
    
    # Split back
    if test_df is not None:
        return df.iloc[:len(train_df)].copy(), df.iloc[len(train_df):].copy(), new_features
    else:
        return df, new_features

if __name__ == '__main__':
    from data import load_data
    import os
    
    # Suppress FutureWarnings from pandas resampling
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    print("Loading data...")
    try:
        # Assumes running from root dir
        df = load_data('data/train.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
        
    transformed_df, generated_cols = add_location_features(df)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY TABLE")
    print("="*60)
    
    # Extract just the newly created non-one-hot features for a cleaner summary
    # Or just group the one-hot features
    one_hots = [c for c in generated_cols if c.startswith('town_') and c not in ['town_year', 'town_flat_type', 'town_freq_enc', 'town_target_enc_cv']]
    summary_cols = [c for c in generated_cols if c not in one_hots]
    
    summary_data = []
    for col in summary_cols:
        dtype = str(transformed_df[col].dtype)
        non_null = transformed_df[col].notnull().sum()
        sample_val = transformed_df[col].dropna().iloc[0] if non_null > 0 else "NaN"
        summary_data.append([col, dtype, non_null, sample_val])
        
    summary_df = pd.DataFrame(summary_data, columns=['Feature Name', 'Data Type', 'Non-Null Count', 'Sample Value'])
    print(summary_df.to_string(index=False))
    
    print(f"\n[Note] Additionally created {len(one_hots)} one-hot encoded 'town' features.")
    print("="*60)
    print("Successfully transformed the dataframe and avoided data leakage.")
