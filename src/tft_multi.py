"""TFT multi-horizon."""

import pandas as pd
import numpy as np
import lightning.pytorch as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


# ============================================================
# 1. Charger et préparer
# ============================================================
df = pd.read_csv("data/AAPL_features.csv", parse_dates=["Date"])
df["time_idx"] = range(len(df))
df["group_id"] = "AAPL"

train = df[df["Date"] <= "2022-12-31"].copy()
val   = df[(df["Date"] >= "2023-01-01") & (df["Date"] <= "2024-12-31")].copy()
test  = df[df["Date"] >= "2025-01-01"].copy()

print("Train :", train.shape)
print("Val   :", val.shape)
print("Test  :", test.shape)

# ============================================================
# 2. Dataset avec max_prediction_length = 20
# ============================================================
MAX_ENCODER = 60
MAX_PREDICT = 20

training = TimeSeriesDataSet(
    train,
    time_idx="time_idx",
    target="target_1d",
    group_ids=["group_id"],
    max_encoder_length=MAX_ENCODER,
    max_prediction_length=MAX_PREDICT,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["log_return", "rsi", "macd", "macd_signal", "macd_diff"],
    target_normalizer=GroupNormalizer(groups=["group_id"]),
)

validation = TimeSeriesDataSet.from_dataset(training, val)
train_dl = training.to_dataloader(train=True, batch_size=64)
val_dl = validation.to_dataloader(train=False, batch_size=64)

# ============================================================
# 3. Modèle TFT
# ============================================================
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
)

n_params = sum(p.numel() for p in tft.parameters())
print(f"\nParamètres : {n_params:,}")

# ============================================================
# 4. Entraînement
# ============================================================
print("\n=== Entraînement ===")
trainer = pl.Trainer(max_epochs=20, gradient_clip_val=0.1, enable_progress_bar=True)
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

trainer.save_checkpoint("data/tft_multi.ckpt")
print("\nModèle sauvegardé : data/tft_multi.ckpt")