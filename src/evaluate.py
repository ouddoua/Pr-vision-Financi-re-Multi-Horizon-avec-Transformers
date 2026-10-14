"""Baseline + LSTM sur les 3 splits."""
import pandas as pd
import numpy as np
import torch
import sys
sys.path.append("src")

from models import load_and_prepare, LSTMModel


# ----- Fonctions métriques -----
def rmse(vrai, predit):
    return np.sqrt(np.mean((vrai - predit) ** 2))

def mae(vrai, predit):
    return np.mean(np.abs(vrai - predit))

def da(vrai, predit):
    return np.mean(np.sign(vrai) == np.sign(predit)) * 100


# ----- Charger les données brutes -----
train = pd.read_csv("data/train.csv")
val   = pd.read_csv("data/val.csv")
test  = pd.read_csv("data/test.csv")

print("Train :", train.shape)
print("Val   :", val.shape)
print("Test  :", test.shape)
print()


# ============================================================
# BASELINE (persistance) sur les 3 splits
# ============================================================
print("========== BASELINE ==========")

vrai = train["target_1d"]
predit = train["log_return"]
print(f"Train : RMSE={rmse(vrai, predit):.5f} | MAE={mae(vrai, predit):.5f} | DA={da(vrai, predit):.2f}%")

vrai = val["target_1d"]
predit = val["log_return"]
print(f"Val   : RMSE={rmse(vrai, predit):.5f} | MAE={mae(vrai, predit):.5f} | DA={da(vrai, predit):.2f}%")

vrai = test["target_1d"]
predit = test["log_return"]
print(f"Test  : RMSE={rmse(vrai, predit):.5f} | MAE={mae(vrai, predit):.5f} | DA={da(vrai, predit):.2f}%")


# ============================================================
# LSTM sur le test
# ============================================================
print()
print("========== LSTM ==========")

# Charger les tenseurs fenêtrés
X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare()

# Charger le modèle entraîné
model = LSTMModel(n_features=X_test.shape[2])
model.load_state_dict(torch.load("data/lstm_model.pt"))
model.eval()

# Prédire sur le test
X_test_t = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_test_t).numpy()

# Le baseline correspondant au MÊME découpage
# (log_return du dernier jour de chaque fenêtre)
y_baseline = test["log_return"].values[60:]

print(f"Baseline (même split) : RMSE={rmse(y_test, y_baseline):.5f} | MAE={mae(y_test, y_baseline):.5f} | DA={da(y_test, y_baseline):.2f}%")
print(f"LSTM                  : RMSE={rmse(y_test, y_pred):.5f} | MAE={mae(y_test, y_pred):.5f} | DA={da(y_test, y_pred):.2f}%")