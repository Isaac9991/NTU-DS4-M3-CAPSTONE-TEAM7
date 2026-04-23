import pandas as pd
import numpy as np

def preprocess_data(df):
    df = df.copy()

    # -------------------
    # 1. Drop columns
    # -------------------
    columns_to_drop = [
        "mid_storey",
        "floor_area_sqft",
        "id",
        "lease_commence_date",
        "bus_stop_name",
        "bus_stop_longitude",
        "bus_stop_latitude",
        "street_name",
        "longitude",
        "latitude"
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    # -------------------
    # 2. Clean categorical columns
    # -------------------
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = df[col].replace(["nan", "none", "null", ""], np.nan)
        df[col] = df[col].fillna("missing")
        df[col] = df[col].astype(str).str.strip().str.lower()

    # df = pd.get_dummies(df, columns=categorical_cols, drop_first=True) #commented out

    # -------------------
    # 3. Numeric cleanup
    # -------------------
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        # convert "1.0 -> 1" style columns
        if (df[col] % 1 == 0).all():
            df[col] = df[col].astype("int64")

    # -------------------
    # 4. Boolean normalization
    # -------------------
    bool_cols = [c for c in num_cols if set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]

    for col in bool_cols:
        df[col] = df[col].astype("int8")

    # -------------------
    # 5. Final safety
    # -------------------
    df = df.fillna(0)

    return df