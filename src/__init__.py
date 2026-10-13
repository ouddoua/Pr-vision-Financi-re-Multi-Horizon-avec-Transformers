import yfinance as yf
import os
import pandas  as pd


app = yf.download("AAPL","2015-09-12")
df = pd.DataFrame(app)
df.to_csv("apple.csv")
print(df)