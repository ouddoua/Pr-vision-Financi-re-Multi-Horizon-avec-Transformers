import requests
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

ticker = "AAPL"

# 1. Récupérer les news NASDAQ (avec dates)
url = f"https://api.nasdaq.com/api/news/topic/articlebysymbol?q={ticker}|STOCKS&offset=0&limit=50"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).json()
articles = r.get("data", {}).get("rows", [])

titres = [a["title"] for a in articles]
dates  = [a.get("created", "") for a in articles]
print(f"{len(titres)} news")

# 2. FinBERT — top_k=None pour avoir les 3 probabilités
clf = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None)
resultats = clf(titres)

# 3. Score continu = P(positive) - P(negative), entre -1 et +1
scores = []
for res in resultats:
    probs = {r["label"].lower(): r["score"] for r in res}
    scores.append(probs["positive"] - probs["negative"])

# 4. Sauvegarder le détail
df = pd.DataFrame({"date": dates, "titre": titres, "score": scores})
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
df.to_csv(f"data/{ticker}_sentiment.csv", index=False)
print(df)

# 5. Agréger par jour
daily = (
    df.groupby(df["date"].dt.date)
    .agg(sentiment_mean=("score", "mean"),
         sentiment_std=("score", "std"),
         news_count=("score", "count"))
    .reset_index()
    .rename(columns={"date": "Date"})
)
daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
daily.to_csv(f"data/{ticker}_sentiment_daily.csv", index=False)
print("\nAgrégé par jour :")
print(daily)

# 6. VISUALISATION — distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["score"], bins=20, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="black", linestyle="--")
axes[0].set_xlabel("Score FinBERT (P(pos) - P(neg))")
axes[0].set_ylabel("Nombre de news")
axes[0].set_title(f"Distribution du sentiment — {ticker}")

pos = (df["score"] > 0.1).sum()
neu = (df["score"].abs() <= 0.1).sum()
neg = (df["score"] < -0.1).sum()
axes[1].pie([pos, neu, neg],
            labels=[f"Positif ({pos})", f"Neutre ({neu})", f"Négatif ({neg})"],
            colors=["green", "grey", "red"], autopct="%1.0f%%")
axes[1].set_title("Répartition des sentiments")

plt.tight_layout()
plt.savefig(f"data/{ticker}_sentiment.png", dpi=120)
plt.show()

# 7. VISUALISATION — zoom sur les 30 derniers jours (prix + news)
prix = pd.read_csv(f"data/{ticker}.csv", parse_dates=["Date"])
prix_recent = prix[prix["Date"] >= df["date"].min() - pd.Timedelta(days=5)]

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(prix_recent["Date"], prix_recent["Close"], color="black", label="Prix", linewidth=2)
ax1.set_ylabel("Prix ($)", color="black")
ax1.tick_params(axis="y", labelcolor="black")

ax2 = ax1.twinx()
colors = ["green" if s > 0 else "red" if s < 0 else "grey" for s in daily["sentiment_mean"]]
ax2.bar(daily["Date"], daily["sentiment_mean"], color=colors, alpha=0.5, width=0.8)
ax2.axhline(0, color="grey", linestyle="--", linewidth=0.7)
ax2.set_ylabel("Sentiment moyen", color="C0")
ax2.tick_params(axis="y", labelcolor="C0")

plt.title(f"{ticker} — Prix vs Sentiment (30 derniers jours)")
fig.tight_layout()
plt.savefig(f"data/{ticker}_sentiment_vs_price.png", dpi=120)
plt.show()

# 8. Stats finales
print("\n--- Statistiques ---")
print(f"Score moyen : {df['score'].mean():.3f}")
print(f"Positives   : {pos} ({pos/len(df)*100:.0f}%)")
print(f"Neutres     : {neu} ({neu/len(df)*100:.0f}%)")
print(f"Négatives   : {neg} ({neg/len(df)*100:.0f}%)")