"""Crée les splits temporels normalisés pour le LSTM."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


SCALED_FEATURES = ["log_return", "macd", "macd_signal", "macd_diff"]
REQUIRED_COLUMNS = SCALED_FEATURES + ["Date", "rsi", "target_1d", "target_5d", "target_20d"]


def prepare_splits(input_path="data/AAPL_features.csv", output_dir="data"):
    data = pd.read_csv(input_path, parse_dates=["Date"])
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Colonnes manquantes dans {input_path}: {missing}")

    data = data.sort_values("Date").reset_index(drop=True)
    masks = {
        "train": data["Date"] <= "2022-12-31",
        "val": data["Date"].between("2023-01-01", "2024-12-31"),
        "test": data["Date"] >= "2025-01-01",
    }
    splits = {name: data.loc[mask].copy() for name, mask in masks.items()}
    empty = [name for name, frame in splits.items() if frame.empty]
    if empty:
        raise ValueError(f"Splits vides: {empty}")

    scaler = StandardScaler()
    scaler.fit(splits["train"][SCALED_FEATURES])
    for frame in splits.values():
        frame.loc[:, SCALED_FEATURES] = scaler.transform(frame[SCALED_FEATURES])
        frame.loc[:, "rsi"] = frame["rsi"] / 100.0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        destination = output_path / f"{name}.csv"
        frame.to_csv(destination, index=False)
        print(
            f"{name}: {len(frame)} lignes, {frame['Date'].min().date()} à "
            f"{frame['Date'].max().date()} -> {destination}"
        )

    print("Scaler ajusté uniquement sur le train.")
    return splits, scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/AAPL_features.csv")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    prepare_splits(args.input, args.output_dir)


if __name__ == "__main__":
    main()