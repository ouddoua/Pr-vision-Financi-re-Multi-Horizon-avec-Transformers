"""Téléchargement des données OHLCV via yfinance."""
import yfinance as yf
import os

def download_data(ticker="AAPL", start="2015-01-01", end="2024-01-01",
                  save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Téléchargement de {ticker} de {start} à {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    filepath = os.path.join(save_dir, f"{ticker}.csv")
    df.to_csv(filepath)
    print(f"Données sauvegardées dans {filepath} ({len(df)} lignes)")
    return df

if __name__ == "__main__":
    download_data()
