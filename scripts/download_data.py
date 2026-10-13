import yfinance as yf
import os
import pandas as pd
from src import feature_engineering as fe

os.makedirs("data", exist_ok=True)

app = yf.download("AAPL", start="2015-01-01")

# 1) Enlever le niveau "Ticker" (AAPL)
app.columns = app.columns.droplevel(1)

# 2) Enlever le nom "Price" qui reste au-dessus
app.columns.name = None

df = pd.DataFrame(app)

df.to_csv("data/AAPL.csv")
features_df = fe.add_technical_indicators(df)
features_df.to_csv("data/AAPL_with_features.csv", index=False)

print(df.head())
print(df.tail())
print(df.shape)
print(df.isna().sum())