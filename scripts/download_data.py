import yfinance as yf
import os
import pandas as pd

os.makedirs("data", exist_ok=True)

app = yf.download("AAPL", start="2015-01-01")

# 1) Enlever le niveau "Ticker" (AAPL)
app.columns = app.columns.droplevel(1)

# 2) Enlever le nom "Price" qui reste au-dessus
app.columns.name = None

df = pd.DataFrame(app)
df.to_csv("data/AAPL.csv")

print(df.head())
print(df.tail())
print(df.shape)