"""Indicateurs techniques et features de sentiment."""
import importlib

import pandas as pd
import numpy as np

def _load_ta_indicator(module_name, class_name):
    """Charge un indicateur Technique Analysis en douceur.

    Cela évite les faux positifs de l'analyseur de code lorsqu'un environnement
    n'a pas encore installé le package `ta`.
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError:
        class MissingIndicator:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    f"Le package 'ta' est requis pour utiliser {class_name}. "
                    "Installez-le avec : pip install ta"
                )

        return MissingIndicator


RSIIndicator = _load_ta_indicator("ta.momentum", "RSIIndicator")
MACD = _load_ta_indicator("ta.trend", "MACD")
BollingerBands = _load_ta_indicator("ta.volatility", "BollingerBands")
OnBalanceVolumeIndicator = _load_ta_indicator("ta.volume", "OnBalanceVolumeIndicator")

def add_technical_indicators(data):
    data = data.copy()

    # RSI
    rsi = RSIIndicator(data['Close'], window=14)
    data['RSI'] = rsi.rsi()

    # MACD
    macd = MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()

    # Bollinger Bands
    bb = BollingerBands(data['Close'], window=20, window_dev=2)
    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()

    # On-Balance Volume
    obv = OnBalanceVolumeIndicator(data['Close'], data['Volume'])
    data['OBV'] = obv.on_balance_volume()

    return data
 
print("Feature engineering module loaded successfully.")
df1=pd.read_csv("data/AAPL.csv")
df1 = add_technical_indicators(df1)
print(df1.head())
