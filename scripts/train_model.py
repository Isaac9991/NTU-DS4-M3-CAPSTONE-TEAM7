import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. HELPER FUNCTIONS ---
def calculate_haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    return c * 6371 

def kfold_target_encode(train, test, column, target):
    train[f'{column}_val'] = np.nan
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train):
        mean_val = train.iloc[train_idx].groupby(column)[target].mean()
        train.loc[train.index[val_idx], f'{column}_val'] = train.loc[train.index[val_idx], column].map(mean_val)
    global_mean = train[target].mean()
    train[f'{column}_val'] = train[f'{column}_val'].fillna(global_mean)
    full_mean = train.groupby(column)[target].mean()
    test[f'{column}_val'] = test[column].map(full_mean).fillna(global_mean)
    return train, test

def preprocess(df):
    # --- 1. GEOSPATIAL BASE ---
    # Central Business District coordinates
    cbd_lat, cbd_lon = 1.2830, 103.8513
    df['dist_to_cbd'] = calculate_haversine(df['Latitude'], df['Longitude'], cbd_lat, cbd_lon)
    
    # --- 2. TIME & MARKET MOMENTUM (2012-2021) ---
    if 'Tranc_YearMonth' in df.columns:
        # Split YYYY-MM into numerical components
        df['year'] = df['Tranc_YearMonth'].str.split('-').str[0].astype(int)
        df['month'] = df['Tranc_YearMonth'].str.split('-').str[1].astype(int)
        
        # Create a continuous time index (e.g., 2015.5 for June 2015)
        df['time_index'] = df['year'] + (df['month'] / 12)
        
        # Policy & Market Shift Flags
        # May 2019: CPF rule changes allowing more flexibility for older flats
        df['is_post_2019_rules'] = (df['time_index'] > 2019.4).astype(int)
        # Mid-2020: The start of the COVID-19 resale price surge
        df['is_covid_surge'] = (df['time_index'] > 2020.5).astype(int)

    # --- 3. REFINED AGE & LEASE LOGIC ---
    # Flat age at the exact time of transaction
    df['flat_age_at_sale'] = df['year'] - df['lease_commence_date']
    
    # Feature A: The MOP Premium (Flats aged 5-9 years)
    # These are often the "hottest" commodities in the market
    df['is_mop_premium'] = ((df['flat_age_at_sale'] >= 5) & (df['flat_age_at_sale'] <= 9)).astype(int)
    
    # Feature B: The Lease Cliff (Flats older than 40 years)
    # Accounts for the drop in value when financing becomes restrictive
    df['is_old_lease_cliff'] = (df['flat_age_at_sale'] > 40).astype(int)
    
    # Stable Age Math (Reverting to the raw version from your high-score run)
    df['age_squared'] = df['flat_age_at_sale'] ** 2
    df['remaining_lease'] = 99 - df['flat_age_at_sale']
    df['lease_log'] = np.log1p(df['remaining_lease'])

    # --- 4. SCHOOLS & CONVENIENCE ---
    if 'cutoff_point' in df.columns:
        df['cutoff_point'] = df['cutoff_point'].fillna(df['cutoff_point'].median())
        
        # Prestige weighted by physical distance to the school
        df['sec_sch_prestige_index'] = (df['cutoff_point']**2) / (df['sec_sch_nearest_dist'] + 0.5)
        
        # CBD Prestige Interaction: Elite schools in central areas carry higher weight
        df['prestige_cbd_interaction'] = (df['cutoff_point']**2) / (df['dist_to_cbd'] + 1)

    # Convenience Friction: Combined penalty for being far from the city AND the train
    # Higher scores = more isolated/less valuable locations
    df['convenience_friction'] = df['dist_to_cbd'] * np.log1p(df['mrt_nearest_distance'])

    # --- 5. CLEANUP & BINARY MAPPING ---
    # Extract postal sector (first 2 digits) to capture neighborhood micro-climates
    df['postal_sector'] = df['postal'].astype(str).str.zfill(6).str[:2]
    
    # Drop high-cardinality or redundant columns to reduce noise
    redundant = ['Tranc_YearMonth', 'storey_range', 'mid', 'full_flat_type', 'address', 'postal']
    df = df.drop(columns=[c for c in redundant if c in df.columns])
    
    # Standardize binary indicators
    binary_cols = ['residential', 'commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Y': 1, 'N': 0}).fillna(0)
            
    # Handle missing school data
    if 'vacancy' in df.columns:
        df['vacancy'] = df['vacancy'].fillna(df['vacancy'].median())
    if 'pri_sch_affiliation' in df.columns:
        df['pri_sch_affiliation'] = df['pri_sch_affiliation'].map({'Y': 1, 'N': 0}).fillna(0)

    return df

# --- 2. DATA LOADING ---
print("Loading and Preprocessing Datasets...")
train = pd.read_csv('train.csv', dtype={'postal': str})
test = pd.read_csv('test.csv', dtype={'postal': str})

# This creates the train_df and test_df you were missing!
train_df = preprocess(train)
test_df = preprocess(test)

# --- 3. TARGET ENCODING ---
print("Applying Target Encoding...")
for col in ['pri_sch_name', 'sec_sch_name', 'postal_sector']:
    train_df, test_df = kfold_target_encode(train_df, test_df, col, 'resale_price')

# --- 4. CATEGORICAL ENCODING ---
print("Applying Categorical Encoding...")
cat_cols = ['town', 'flat_type', 'flat_model', 'planning_area', 'mrt_name']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# --- 5. TRAINING ---
print("Starting XGBoost Training...")
X = train_df.drop(columns=['id', 'resale_price'])
y = np.log1p(train_df['resale_price'])
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(X[numerical_features], y, test_size=0.15, random_state=42)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=12000,      # Increase trees
    learning_rate=0.005,     # Slow down learning for more precision
    max_depth=10,            # Go back to 10
    subsample=0.85,          # See more data per tree
    colsample_bytree=0.5,    # Force more feature variety
    reg_lambda=12.0,         # Stronger L2 to stop outliers from ruining RMSE
    reg_alpha=1.5,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=200
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

# --- 6. EVALUATION ---
val_preds_log = model.predict(X_val)
val_preds_log = np.clip(val_preds_log, y.min(), y.max()) # Guardrail

val_preds = np.expm1(val_preds_log)
y_val_actual = np.expm1(y_val)

print(f"\n--- Enhanced Validation Results ---")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_val_actual, val_preds)):,.2f}")
print(f"MAE:  ${mean_absolute_error(y_val_actual, val_preds):.2f}")
print(f"R2 Score: {r2_score(y_val_actual, val_preds):.4f}")

# --- 7. SUBMISSION ---
test_preds_log = model.predict(test_df[numerical_features])
test_preds_log = np.clip(test_preds_log, y.min(), y.max())
test_preds = np.expm1(test_preds_log)
pd.DataFrame({'id': test['id'], 'resale_price': test_preds}).to_csv('final_stable_submission.csv', index=False)
print("✅ Done! Submission file 'final_stable_submission.csv' is ready.")