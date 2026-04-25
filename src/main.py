from data import load_data
from feature_eng import SchoolFeature, feature_engineering_pipeline, preprocess as feature_preprocess
from preprocess import preprocess_data
from models.xgboost import get_xgboost_model
from models.lightgbm import get_lightgbm_model
from encoding import TargetEncoder
from train import train_model

from collections import defaultdict

def main():

    #get data
    print("Loading data...")
    train_df = load_data("data/train.csv")
    test_df = load_data("data/test.csv")

    #preprocess data
    print("Preprocessing data...")
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    print("Feature preprocessing...")
    train_df = feature_preprocess(train_df)
    test_df = feature_preprocess(test_df)

    print("Feature engineering...")

    school_fe = SchoolFeature()

    # fit only on train
    school_fe.fit(train_df)

    # feature engineering
    train_df = feature_engineering_pipeline(train_df, school_fe)
    test_df = feature_engineering_pipeline(test_df, school_fe)

    #encoding: target encoding first, then label encoding for remaining categoricals

    print("Target encoding...")

    cols_to_encode = [
        "town",
        "flat_model",
        "planning_area",
        "mrt_name"
    ]

    te = TargetEncoder(cols=cols_to_encode)

    train_df = te.fit_transform(train_df, target="resale_price")
    test_df = te.transform(test_df)

    # drop original categorical columns

    print("Cleaning remaining object columns...")

    train_df = train_df.drop(columns=train_df.select_dtypes(include=["object"]).columns)
    test_df = test_df.drop(columns=test_df.select_dtypes(include=["object"]).columns)


    #get model
    models = {
        "xgboost": get_xgboost_model()
        # "lightgbm": get_lightgbm_model()
        }
    
    results = {}
    trained_models = {}

    #train model
    print("Training model...")
    for name, model in models.items():

        print(f"Training {name} model...")

        trained_model, rmse = train_model(train_df, model)

        trained_models[name] = trained_model
        results[name] = rmse

        print(f"{name} RMSE: {rmse:.4f}")

    

if __name__ == "__main__":
    main()