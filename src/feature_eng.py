import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#TODO: add more features here

# def add_time_features(df):
#     df = df.copy()

#     df["Tranc_YearMonth"] = pd.to_datetime(df["Tranc_YearMonth"])
#     df["year"] = df["Tranc_YearMonth"].dt.year
#     df["month"] = df["Tranc_YearMonth"].dt.month

#     return df.drop(columns=["Tranc_YearMonth"])

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

    return df