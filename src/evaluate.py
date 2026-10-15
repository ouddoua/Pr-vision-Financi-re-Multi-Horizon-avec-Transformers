"""Baseline + LSTM multi-horizon sur le test."""
import argparse
import pandas as pd
import numpy as np
import torch
import sys
sys.path.append("src")

from models import load_and_prepare, LSTMModel, L


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default="data/lstm_model.pt")
args = parser.parse_args()


def rmse(vrai, predit):
    return np.sqrt(np.mean((vrai - predit) ** 2))

def mae(vrai, predit):
    return np.mean(np.abs(vrai - predit))

def da(vrai, predit):
    return np.mean(np.sign(vrai) == np.sign(predit)) * 100


# ----- Charger les données -----
test = pd.read_csv("data/test.csv")
X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare()

print("Test shapes :")
print("  y_test :", y_test.shape)
print()


# ============================================================
# Charger le modèle
# ============================================================
model = LSTMModel(n_features=X_test.shape[2])
model.load_state_dict(torch.load(args.model, map_location="cpu"))
model.eval()

X_test_t = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_test_t).numpy()


# ============================================================
# Baseline persistance multi-horizon
# ============================================================
# Pour horizon h, on prédit la somme des h derniers log_returns
# (dans la fenêtre de 60 jours qui précède)

close = test["Close"].values

# On aligne : la fenêtre commence à l'index L
baseline_1d  = np.array([np.log(close[i-1] / close[i-2]) for i in range(L, len(test))])
baseline_5d  = np.array([np.log(close[i-1] / close[i-6]) for i in range(L, len(test))])
baseline_20d = np.array([np.log(close[i-1] / close[i-21]) for i in range(L, len(test))])

baseline = np.column_stack([baseline_1d, baseline_5d, baseline_20d])


# ============================================================
# Comparaison horizon par horizon
# ============================================================
horizons = ["1 jour", "5 jours", "20 jours"]

for i, h in enumerate(horizons):
    print(f"========== Horizon : {h} ==========")
    y_vrai = y_test[:, i]

    # Baseline
    print(f"  Baseline : RMSE={rmse(y_vrai, baseline[:, i]):.5f} | "
          f"MAE={mae(y_vrai, baseline[:, i]):.5f} | "
          f"DA={da(y_vrai, baseline[:, i]):.2f}%")

    # LSTM
    print(f"  LSTM     : RMSE={rmse(y_vrai, y_pred[:, i]):.5f} | "
          f"MAE={mae(y_vrai, y_pred[:, i]):.5f} | "
          f"DA={da(y_vrai, y_pred[:, i]):.2f}%")
    print()