"""Préparation des données et modèles."""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


FEATURES = [
    "log_return", "rsi", "macd", "macd_signal", "macd_diff"
]
TARGET = "target_1d"
L = 60


def make_windows(df, L, feature_cols, target_col):
    data_X = df[feature_cols].values
    data_y = df[target_col].values

    X = []
    y = []

    for i in range(L, len(df)):
        X.append(data_X[i - L:i])
        y.append(data_y[i])

    return np.array(X), np.array(y)


def load_and_prepare():
    train = pd.read_csv("data/train.csv")
    val   = pd.read_csv("data/val.csv")
    test  = pd.read_csv("data/test.csv")

    X_train, y_train = make_windows(train, L, FEATURES, TARGET)
    X_val,   y_val   = make_windows(val,   L, FEATURES, TARGET)
    X_test,  y_test  = make_windows(test,  L, FEATURES, TARGET)

    return X_train, y_train, X_val, y_val, X_test, y_test


class LSTMModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        sorties, _ = self.lstm(x)
        dernier = sorties[:, -1, :]
        prediction = self.fc(dernier)
        return prediction.squeeze(-1)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare()

    print("X_train :", X_train.shape, "| y_train :", y_train.shape)
    print("X_val   :", X_val.shape,   "| y_val   :", y_val.shape)
    print("X_test  :", X_test.shape,  "| y_test  :", y_test.shape)

    model = LSTMModel(n_features=X_train.shape[2])
    print()
    print(model)