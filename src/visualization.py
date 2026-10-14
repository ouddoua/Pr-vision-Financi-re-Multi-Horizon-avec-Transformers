import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/AAPL_features.csv", parse_dates=["Date"], index_col="Date")
print("Colonnes :", list(df.columns))
print("Shape    :", df.shape)
print("De ", df.index.min().date(), "à", df.index.max().date())

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# --- 1. Prix + Bollinger ---
ax = axes[0]
ax.plot(df.index, df["Close"], label="Close", color="black", linewidth=1)
ax.plot(df.index, df["bb_upper"], label="Bollinger upper", color="red", linestyle="--", alpha=0.6)
ax.plot(df.index, df["bb_lower"], label="Bollinger lower", color="red", linestyle="--", alpha=0.6)
ax.plot(df.index, df["bb_middle"], label="Bollinger middle", color="orange", linewidth=0.8)
ax.set_title("Prix + Bollinger Bands")
ax.set_ylabel("Prix ($)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# --- 2. RSI ---
ax = axes[1]
ax.plot(df.index, df["rsi"], label="RSI(14)", color="purple", linewidth=1)
ax.axhline(70, color="red", linestyle="--", alpha=0.7, label="Surachat (70)")
ax.axhline(30, color="green", linestyle="--", alpha=0.7, label="Survente (30)")
ax.set_title("RSI (Relative Strength Index)")
ax.set_ylabel("RSI")
ax.set_ylim(0, 100)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# --- 3. MACD ---
ax = axes[2]
ax.plot(df.index, df["macd"], label="MACD", color="blue", linewidth=1)
ax.plot(df.index, df["macd_signal"], label="Signal", color="orange", linewidth=1)
ax.bar(df.index, df["macd_diff"], label="Histogramme", color="gray", alpha=0.5, width=1.0)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("MACD")
ax.set_ylabel("Valeur")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# --- 4. Volume ---
ax = axes[3]
ax.bar(df.index, df["Volume"], color="steelblue", width=1.0)
ax.set_title("Volume")
ax.set_ylabel("Volume")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/features_overview.png", dpi=100, bbox_inches="tight")
print("Graphique sauvegardé dans results/features_overview.png")
plt.show()