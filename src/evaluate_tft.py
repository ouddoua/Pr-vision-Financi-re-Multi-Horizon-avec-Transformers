"""Évaluation du TFT sur le test."""

import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer


def rmse(vrai, predit):
    return np.sqrt(np.mean((vrai - predit) ** 2))

def mae(vrai, predit):
    return np.mean(np.abs(vrai - predit))

def da(vrai, predit):
    return np.mean(np.sign(vrai) == np.sign(predit)) * 100


# 1. Recharger les données
df = pd.read_csv("data/AAPL_features.csv", parse_dates=["Date"])
df["time_idx"] = range(len(df))
df["group_id"] = "AAPL"

train = df[df["Date"] <= "2022-12-31"].copy()
test  = df[df["Date"] >= "2025-01-01"].copy()

# 2. Recréer le dataset d'entraînement
training = TimeSeriesDataSet(
    train,
    time_idx="time_idx",
    target="target_1d",
    group_ids=["group_id"],
    max_encoder_length=60,
    max_prediction_length=1,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["log_return", "rsi", "macd", "macd_signal", "macd_diff"],
)

# 3. Dataset test SANS predict=True
testing = TimeSeriesDataSet.from_dataset(training, test, predict=False)
test_dl = testing.to_dataloader(train=False, batch_size=64)

# 4. Charger le modèle
tft = TemporalFusionTransformer.load_from_checkpoint("data/tft_model.ckpt")
tft.eval()
print("Modèle TFT chargé.")

# 5. Prédire
predictions = tft.predict(test_dl, mode="prediction", return_x=False)
y_pred = predictions.numpy().flatten()

# 6. Vraies valeurs
y_true = np.concatenate([y[0].numpy() for x, y in test_dl]).flatten()

print("y_pred :", y_pred.shape)
print("y_true :", y_true.shape)

# 7. Métriques
print()
print("========== TFT sur test ==========")
print(f"RMSE : {rmse(y_true, y_pred):.5f}")
print(f"MAE  : {mae(y_true, y_pred):.5f}")
print(f"DA   : {da(y_true, y_pred):.2f}%")