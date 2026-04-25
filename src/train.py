import pandas as pd
import numpy as np
from models.xgboost import get_xgboost_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model(df, model):
    """
    Train xgboost regression model on the training data.

    """
    X = df.drop(columns=["resale_price"])
    Y = np.log1p(df["resale_price"])
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    #train model
    model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)

    #evaluate model
    Y_pred = model.predict(X_val)

    #convert back to original scale
    Y_pred = np.expm1(Y_pred) #predicted values
    Y_val_true = np.expm1(Y_val) #actual values from train.csv

    #compare accuracy of predicted values with actual values
    rmse = np.sqrt(mean_squared_error(Y_val_true, Y_pred))

    r2 = r2_score(Y_val_true, Y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    return model, rmse
