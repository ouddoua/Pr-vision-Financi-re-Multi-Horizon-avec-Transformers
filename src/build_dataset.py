"""Fusionne features techniques + sentiment en un seul dataset."""
import pandas as pd

TICKER = "AAPL"

# 1. Charger features techniques
df = pd.read_csv(f"data/{TICKER}_features.csv", parse_dates=["Date"])

# 2. Charger sentiment quotidien
sent = pd.read_csv(f"data/{TICKER}_sentiment_daily.csv", parse_dates=["Date"])

# 3. Fusion sur la date
df = df.merge(sent, on="Date", how="left")

# 4. Jours sans news = neutre
df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)
df["sentiment_std"]  = df["sentiment_std"].fillna(0.0)
df["news_count"]     = df["news_count"].fillna(0.0)

# 5. Sauvegarder
df.to_csv(f"data/{TICKER}_full.csv", index=False)
print(f"Dataset fusionné : {df.shape}")
print(f"Colonnes : {list(df.columns)}")
print(df[["Date", "Close", "rsi", "sentiment_mean", "target_1d"]].tail())