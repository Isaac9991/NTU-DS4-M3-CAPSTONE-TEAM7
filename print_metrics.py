import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("output/oof_predictions.csv")
rmse = np.sqrt(mean_squared_error(df["actual_price"], df["predicted_price"]))
r2 = r2_score(df["actual_price"], df["predicted_price"])

print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.4f}")
