import pandas as pd
import numpy as np
import ta

# Charger les données propres (celles qu'on a sauvées après processing)
df = pd.read_csv("data/AAPl_clean.csv", parse_dates=["Date"], index_col="Date")
print("Shape initial :", df.shape)

# Log-return
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))


# RSI
df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

# MACD
macd = ta.trend.MACD(df["Close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_diff"] = macd.macd_diff()

# Bollinger Bands
bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
df["bb_upper"] = bb.bollinger_hband()
df["bb_middle"] = bb.bollinger_mavg()
df["bb_lower"] = bb.bollinger_lband()

df["target_1d"]  = np.log(df["Close"].shift(-1)  / df["Close"])
df["target_5d"]  = np.log(df["Close"].shift(-5)  / df["Close"])
df["target_20d"] = np.log(df["Close"].shift(-20) / df["Close"])

print("Shape après features + targets :", df.shape)
print(df[["Close", "rsi", "macd", "bb_upper", "log_return"]].tail())
df = df.dropna()

print("Après nettoyage :", df.shape)
df.to_csv("data/AAPL_features.csv")

print()
print("Colonnes :", list(df.columns))
print("Période  :", df.index.min().date(), "→", df.index.max().date())
print("Sauvegardé dans data/AAPL_features.csv")
