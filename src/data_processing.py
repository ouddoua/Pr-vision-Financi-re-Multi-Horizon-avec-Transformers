"""Split temporel + normalisation."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


COLS_TO_SCALE = [
    "Close", "High", "Low", "Open", "Volume",
    "macd", "macd_signal", "macd_diff",
    "bb_upper", "bb_middle", "bb_lower"
]


def load_features(file_path):
    return pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")


def split_time(df):
    train = df.loc["2015-02-20":"2022-12-31"].copy()
    val   = df.loc["2023-01-01":"2024-12-31"].copy()
    test  = df.loc["2025-01-01":].copy()
    return train, val, test


def normalize_data(train, val, test):
    scaler = StandardScaler()
    scaler.fit(train[COLS_TO_SCALE])

    train[COLS_TO_SCALE] = scaler.transform(train[COLS_TO_SCALE])
    val[COLS_TO_SCALE]   = scaler.transform(val[COLS_TO_SCALE])
    test[COLS_TO_SCALE]  = scaler.transform(test[COLS_TO_SCALE])

    train["rsi"] = train["rsi"] / 100
    val["rsi"]   = val["rsi"]   / 100
    test["rsi"]  = test["rsi"]  / 100

    return train, val, test, scaler


if __name__ == "__main__":
    df = load_features("data/AAPL_features.csv")
    print("Shape total :", df.shape)
    print("Période     :", df.index.min().date(), "→", df.index.max().date())

    train, val, test = split_time(df)
    print()
    print("Train :", train.shape, "|", train.index.min().date(), "→", train.index.max().date())
    print("Val   :", val.shape,   "|", val.index.min().date(),   "→", val.index.max().date())
    print("Test  :", test.shape,  "|", test.index.min().date(),  "→", test.index.max().date())

    train, val, test, scaler = normalize_data(train, val, test)

    print()
    print("--- Stats après normalisation ---")
    print("Train (moyennes ≈ 0, écarts ≈ 1) :")
    print(train[COLS_TO_SCALE].describe().loc[["mean", "std"]].round(3))
    print()
    print("Test (moyennes ≠ 0, c'est NORMAL) :")
    print(test[COLS_TO_SCALE].describe().loc[["mean", "std"]].round(3))

    train.to_csv("data/train.csv")
    val.to_csv("data/val.csv")
    test.to_csv("data/test.csv")

    print()
    print("Sauvegardés :")
    print("  data/train.csv :", train.shape)
    print("  data/val.csv   :", val.shape)
    print("  data/test.csv  :", test.shape)