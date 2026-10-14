"""Évaluation du TFT multi-horizon."""

import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer


def rmse(vrai, predit):
    return np.sqrt(np.mean((vrai - predit) ** 2))

def mae(vrai, predit):
    return np.mean(np.abs(vrai - predit))

def da(vrai, predit):
    return np.mean(np.sign(vrai) == np.sign(predit)) * 100


# ============================================================
# 1. Recharger les données
# ============================================================
df = pd.read_csv("data/AAPL_features.csv", parse_dates=["Date"])
df["time_idx"] = range(len(df))
df["group_id"] = "AAPL"

train = df[df["Date"] <= "2022-12-31"].copy()
test  = df[df["Date"] >= "2025-01-01"].copy()

training = TimeSeriesDataSet(
    train,
    time_idx="time_idx",
    target="target_1d",
    group_ids=["group_id"],
    max_encoder_length=60,
    max_prediction_length=20,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["log_return", "rsi", "macd", "macd_signal", "macd_diff"],
)

testing = TimeSeriesDataSet.from_dataset(training, test)
test_dl = testing.to_dataloader(train=False, batch_size=64)


# ============================================================
# 2. Charger le modèle
# ============================================================
tft = TemporalFusionTransformer.load_from_checkpoint("data/tft_multi.ckpt")
tft.eval()
print("Modèle TFT multi-horizon chargé.")


# ============================================================
# 3. Prédire
# ============================================================
predictions = tft.predict(test_dl, mode="prediction", return_x=False)

# predictions a la forme (N, 20)
print("predictions shape :", predictions.shape)

y_pred_all = predictions.numpy()
y_true_all = np.concatenate([y[0].numpy() for x, y in test_dl])


# ============================================================
# 4. Construire les 3 horizons
# ============================================================
y_pred_1d  = y_pred_all[:, 0]
y_pred_5d  = y_pred_all[:, :5].sum(axis=1)
y_pred_20d = y_pred_all[:, :20].sum(axis=1)

y_true_1d  = y_true_all[:, 0]
y_true_5d  = y_true_all[:, :5].sum(axis=1)
y_true_20d = y_true_all[:, :20].sum(axis=1)

print("y_pred_1d  :", y_pred_1d.shape)
print("y_pred_5d  :", y_pred_5d.shape)
print("y_pred_20d :", y_pred_20d.shape)


# ============================================================
# 5. Métriques par horizon
# ============================================================
horizons = [
    ("1 jour",  y_true_1d,  y_pred_1d),
    ("5 jours", y_true_5d,  y_pred_5d),
    ("20 jours", y_true_20d, y_pred_20d),
]

for nom, vrai, predit in horizons:
    print(f"========== Horizon : {nom} ==========")
    print(f"  RMSE : {rmse(vrai, predit):.5f}")
    print(f"  MAE  : {mae(vrai, predit):.5f}")
    print(f"  DA   : {da(vrai, predit):.2f}%")
    print()