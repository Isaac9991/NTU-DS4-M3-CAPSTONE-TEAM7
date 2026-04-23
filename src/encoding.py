from sklearn.model_selection import KFold
import numpy as np

class TargetEncoder:
    def __init__(self, cols=None, n_splits=5):
        self.cols = cols
        self.n_splits = n_splits
        self.global_mean = None
        self.maps = {}

    def fit_transform(self, train_df, target):

        train_df = train_df.copy()
        
        if self.cols is None:
            self.cols = train_df.select_dtypes(include=["object"]).columns.tolist()

        self.global_mean = train_df[target].mean()

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for col in self.cols:
            train_df[f"{col}_te"] = np.nan

            for train_idx, val_idx in kf.split(train_df):
                mean_map = train_df.iloc[train_idx].groupby(col)[target].mean()
                train_df.loc[train_df.index[val_idx], f"{col}_te"] = (
                    train_df.loc[train_df.index[val_idx], col].map(mean_map)
                )

            train_df[f"{col}_te"] = train_df[f"{col}_te"].fillna(self.global_mean)

            self.maps[col] = train_df.groupby(col)[target].mean()

        return train_df

    def transform(self, df):
        df = df.copy()

        for col in self.cols:
            df[f"{col}_te"] = df[col].map(self.maps[col]).fillna(self.global_mean)

        return df