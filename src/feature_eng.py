import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#TODO: add more features here

def calculate_haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371


def add_amenity_scores(df):
    df = df.copy()

    mall_500 = df["Mall_Within_500m"].fillna(0)
    mall_1k  = df["Mall_Within_1km"].fillna(0)
    mall_2k  = df["Mall_Within_2km"].fillna(0)

    hawker_500 = df["Hawker_Within_500m"].fillna(0)
    hawker_1k  = df["Hawker_Within_1km"].fillna(0)
    hawker_2k  = df["Hawker_Within_2km"].fillna(0)

    df["mall_score"] = 3 * mall_500 + 2 * mall_1k + 1 * mall_2k
    df["hawker_score"] = 3 * hawker_500 + 2 * hawker_1k + 1 * hawker_2k

    df["mall_distance_score"] = -df["Mall_Nearest_Distance"].fillna(df["Mall_Nearest_Distance"].median())
    df["hawker_distance_score"] = -df["Hawker_Nearest_Distance"].fillna(df["Hawker_Nearest_Distance"].median())

    return df


class SchoolFeature:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df):
        cols = [
            "pri_sch_nearest_distance",
            "cutoff_point",
            "vacancy",
            "pri_sch_affiliation"
        ]

        self.scaler.fit(df[cols])
        self.fitted = True
        return self

    def transform(self, df):
        df = df.copy()

        cols = [
            "pri_sch_nearest_distance",
            "cutoff_point",
            "vacancy",
            "pri_sch_affiliation"
        ]

        scaled = self.scaler.transform(df[cols])

        df["school_score"] = (
            -scaled[:, 0]
            + scaled[:, 1]
            + scaled[:, 2]
            + scaled[:, 3]
        )

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
    

def feature_engineering_pipeline(df, school_fe):
    df = df.copy()

    # 1. Amenities (function)
    df = add_amenity_scores(df)

    # 2. School (class)
    df = school_fe.transform(df)

    # 3. Additional interaction features
    if "mall_score" in df.columns and "hawker_score" in df.columns:
        df["total_amenity_score"] = df["mall_score"] + df["hawker_score"]
    
    # 4. Combined convenience score
    if {"mall_distance_score", "hawker_distance_score", "school_score"}.issubset(df.columns):
        df["combined_convenience"] = (
            df["mall_distance_score"] + 
            df["hawker_distance_score"] + 
            df["school_score"]
        )
    
    # 5. Log transformation of age for better distribution
    if "flat_age_at_sale" in df.columns:
        df["age_log"] = np.log1p(df["flat_age_at_sale"])
    
    # 6. Interaction: floor area with remaining lease
    if {"floor_area_sqm", "remaining_lease"}.issubset(df.columns):
        df["area_lease_interaction"] = df["floor_area_sqm"] * np.log1p(df["remaining_lease"])

    return df

def preprocess(df):
    df = df.copy()
    cbd_lat, cbd_lon = 1.2830, 103.8513
    if {'Latitude', 'Longitude'}.issubset(df.columns):
        df['dist_to_cbd'] = calculate_haversine(df['Latitude'], df['Longitude'], cbd_lat, cbd_lon)
    else:
        df['dist_to_cbd'] = np.nan
    
    # --- 1. TIME & MARKET CONTEXT ---
    if 'Tranc_YearMonth' in df.columns:
        df['year'] = df['Tranc_YearMonth'].str.split('-').str[0].astype(int)
        df['month'] = df['Tranc_YearMonth'].str.split('-').str[1].astype(int)
        df['time_index'] = df['year'] + (df['month'] / 12)
        df['is_post_2019_rules'] = (df['time_index'] > 2019.4).astype(int)
        df['is_covid_surge'] = (df['time_index'] > 2020.5).astype(int)
    else:
        df['year'] = np.nan
        df['month'] = np.nan
        df['time_index'] = np.nan
        df['is_post_2019_rules'] = 0
        df['is_covid_surge'] = 0

    # --- 2. DYNAMIC LEASE & AGE ---
    if 'lease_commence_date' in df.columns:
        df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')
        df['flat_age_at_sale'] = df['year'] - df['lease_commence_date']
    else:
        df['flat_age_at_sale'] = np.nan

    df['age_squared'] = df['flat_age_at_sale'] ** 2
    df['is_mop_premium'] = ((df['flat_age_at_sale'] >= 5) & (df['flat_age_at_sale'] <= 9)).fillna(False).astype(int)
    df['remaining_lease'] = 99 - df['flat_age_at_sale']
    df['lease_log'] = np.log1p(df['remaining_lease'].clip(lower=0))

    # --- 3. SCHOOLS & PRESTIGE ---
    if 'cutoff_point' in df.columns:
        df['cutoff_point'] = df['cutoff_point'].fillna(df['cutoff_point'].median())
        if 'dist_to_cbd' in df.columns:
            df['prestige_cbd_interaction'] = (df['cutoff_point'] ** 2) / (df['dist_to_cbd'] + 1)
        if 'sec_sch_nearest_dist' in df.columns:
            df['sec_sch_prestige_index'] = (df['cutoff_point'] ** 2) / (df['sec_sch_nearest_dist'] + 0.5)

    if 'vacancy' in df.columns:
        df['vacancy'] = df['vacancy'].fillna(df['vacancy'].median())
    if 'pri_sch_affiliation' in df.columns:
        df['pri_sch_affiliation'] = df['pri_sch_affiliation'].map({'Y': 1, 'N': 0, 1: 1, 0: 0}).fillna(0)

    # --- 4. PROJECT PREMIUM FEATURES ---
    if {'dist_to_cbd', 'mrt_nearest_distance'}.issubset(df.columns):
        df['convenience_friction'] = df['dist_to_cbd'] * np.log1p(df['mrt_nearest_distance'])

    # --- 5. CLEANUP & EXPLICIT KEEP ---
    if 'postal' in df.columns:
        df['postal_sector'] = df['postal'].astype(str).str.zfill(6).str[:2]

    # We drop these as originals but keep their encoded versions/derived features
    redundant = ['Tranc_YearMonth', 'storey_range', 'mid', 'full_flat_type', 'address', 'postal']
    df = df.drop(columns=[c for c in redundant if c in df.columns])
    
    binary_cols = ['residential', 'commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Y': 1, 'N': 0, 1: 1, 0: 0}).fillna(0)
            
    # Handle socio-economic NaNs (missing usually means zero in these columns)
    if 'exec_sold' in df.columns:
        df['exec_sold'] = df['exec_sold'].fillna(0)
        
    return df