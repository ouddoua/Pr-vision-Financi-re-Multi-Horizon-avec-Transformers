import pandas as pd
import ta

# Charger les données propres (celles qu'on a sauvées après processing)
df = pd.read_csv("data/AAPl_clean.csv", parse_dates=["Date"], index_col="Date")

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

# Log-return (on le garde, car ta ne le fait pas)
df["log_return"] = (df["Close"] / df["Close"].shift(1)).apply(lambda x: __import__('numpy').log(x))

print(df[["Close", "rsi", "macd", "bb_upper", "log_return"]].tail())

df = df.dropna()
print(df.head())
df.to_csv("data/AAPL_features.csv")
