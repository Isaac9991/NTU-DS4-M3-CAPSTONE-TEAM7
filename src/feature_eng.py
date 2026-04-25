import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def preprocess_data(df, is_train=True):
    """
    Preprocess the dataset by extracting features, dropping duplicate or unneeded columns,
    and converting object types to categorical for LightGBM.
    """
    df = df.copy()
    
    # 1. Feature Engineering: Temporal
    if 'Tranc_Year' in df.columns and 'Tranc_Month' in df.columns:
        df['transaction_time'] = df['Tranc_Year'] + (df['Tranc_Month'] - 1) / 12.0
        
    if 'Tranc_Year' in df.columns and 'lease_commence_date' in df.columns:
        df['remaining_lease'] = 99 - (df['Tranc_Year'] - df['lease_commence_date'])

    # 2. Feature Engineering: Spatial CBD Distance
    # CBD coordinates (Raffles Place) -> ~1.2830, 103.8513
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df['CBD_Dist'] = haversine_distance(df['Latitude'], df['Longitude'], 1.2830, 103.8513)
        
    # 3. Aggregate Density Amenities
    if 'Mall_Within_1km' in df.columns and 'Hawker_Within_1km' in df.columns:
        malls = df['Mall_Within_1km'].fillna(0)
        hawkers = df['Hawker_Within_1km'].fillna(0)
        df['amenity_density'] = malls + hawkers

    # Distance/Area features
    if 'floor_area_sqm' in df.columns:
        df['log_floor_area'] = np.log1p(df['floor_area_sqm'])
        
    if 'mrt_nearest_distance' in df.columns:
        df['log_mrt_dist'] = np.log1p(df['mrt_nearest_distance'])
        
    if 'CBD_Dist' in df.columns:
        df['log_cbd_dist'] = np.log1p(df['CBD_Dist'])
        
    if 'mid_storey' in df.columns and 'max_floor_lvl' in df.columns:
        df['relative_height'] = df['mid_storey'] / (df['max_floor_lvl'] + 1)
        
    if 'hdb_age' in df.columns:
        df['age_squared'] = df['hdb_age'] ** 2
        
    # 4. Drop redundant/highly correlated or unneeded ID columns
    # We NO LONGER drop 'mid_storey'. We retain 'mrt_name' as high-value categorical
    # We retain 'block', 'street_name', 'postal' etc for CatBoost
    cols_to_drop = ['floor_area_sqft', 'lease_commence_date', 'Tranc_YearMonth',
                    'address', 'full_flat_type']
    
    if 'id' in df.columns and is_train:
        cols_to_drop.append('id') # Keep ID in test set for submission
        
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # 5. Convert remaining object columns to string for CatBoost
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].astype(str)
        # Handle nan strings
        df[col] = df[col].replace('nan', 'Unknown')
        
    # 6. Separate target if parsing training data
    target = None
    if is_train and 'resale_price' in df.columns:
        target = df['resale_price']
        df = df.drop(columns=['resale_price'])
        
    return df, target
