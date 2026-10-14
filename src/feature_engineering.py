import pandas as pd
import numpy as np
import ta

# 1. Charger le brut
df = pd.read_csv("data/AAPL.csv", parse_dates=["Date"], index_col="Date")
print("Shape initial :", df.shape)

# 2. Log-return (variation passée)
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

# 3. RSI
df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

# 4. MACD
macd = ta.trend.MACD(df["Close"])
df["macd"]        = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_diff"]   = macd.macd_diff()

# 5. Bollinger
bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
df["bb_upper"]  = bb.bollinger_hband()
df["bb_middle"] = bb.bollinger_mavg()
df["bb_lower"]  = bb.bollinger_lband()

# 6. Targets
df["target_1d"]  = np.log(df["Close"].shift(-1)  / df["Close"])
df["target_5d"]  = np.log(df["Close"].shift(-5)  / df["Close"])
df["target_20d"] = np.log(df["Close"].shift(-20) / df["Close"])

print("Après features + targets :", df.shape)

# 7. Nettoyage
df = df.dropna(subset=["target_1d", "target_5d", "target_20d"])
df = df.dropna()

print("Après nettoyage :", df.shape)

# 8. Sauvegarder
df.to_csv("data/AAPL_features.csv")
print("Sauvegardé : data/AAPL_features.csv")