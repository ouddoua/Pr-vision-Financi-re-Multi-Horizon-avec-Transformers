"""Temporal Fusion Transformer."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import lightning.pytorch as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="data/tft_model.ckpt")
args = parser.parse_args()


# ============================================================
# 1. Charger les features (NON normalisées)
# ============================================================
df = pd.read_csv("data/AAPL_features.csv", parse_dates=["Date"])
print("Shape total :", df.shape)

# Ajouter time_idx (0, 1, 2, ...)
df["time_idx"] = range(len(df))

# Ajouter group_id
df["group_id"] = "AAPL"

# ============================================================
# 2. Split par dates
# ============================================================
train = df[df["Date"] <= "2022-12-31"].copy()
val   = df[(df["Date"] >= "2023-01-01") & (df["Date"] <= "2024-12-31")].copy()
test  = df[df["Date"] >= "2025-01-01"].copy()

print("Train :", train.shape)
print("Val   :", val.shape)
print("Test  :", test.shape)

# ============================================================
# 3. Créer le dataset d'entraînement
# ============================================================
MAX_ENCODER = 60     # 60 jours d'historique
MAX_PREDICT = 1      # prédire 1 jour

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

# Créer val et test avec les mêmes paramètres
validation = TimeSeriesDataSet.from_dataset(training, val, predict=True, stop_randomization=True)
testing = TimeSeriesDataSet.from_dataset(training, test, predict=True, stop_randomization=True)

# ============================================================
# 4. Créer les dataloaders
# ============================================================
train_dl = training.to_dataloader(train=True, batch_size=64)
val_dl = validation.to_dataloader(train=False, batch_size=64)
test_dl = testing.to_dataloader(train=False, batch_size=64)

# ============================================================
# 5. Créer le modèle TFT
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
print(f"\nModèle TFT créé. Paramètres : {n_params:,}")

# ============================================================
# 6. Entraîner
# ============================================================
print("\n=== Entraînement ===")
trainer = pl.Trainer(max_epochs=20, gradient_clip_val=0.1, enable_progress_bar=True)
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

# Sauvegarder
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
trainer.save_checkpoint(output_path)
print(f"\nModèle sauvegardé : {output_path}")