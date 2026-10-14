"""Visualisation de l'attention du TFT."""

import pandas as pd
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer


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
    max_prediction_length=1,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["log_return", "rsi", "macd", "macd_signal", "macd_diff"],
)

testing = TimeSeriesDataSet.from_dataset(training, test, predict=False)
test_dl = testing.to_dataloader(train=False, batch_size=64)


# ============================================================
# 2. Charger le TFT
# ============================================================
tft = TemporalFusionTransformer.load_from_checkpoint("data/tft_model.ckpt")
tft.eval()
print("Modèle TFT chargé.")


# ============================================================
# 3. Prédiction en mode "raw"
# ============================================================
resultat = tft.predict(test_dl, mode="raw", return_x=True)

raw_predictions = resultat[0]
x = resultat[1]


# ============================================================
# 4. Interpréter et sauvegarder toutes les figures
# ============================================================
interpretation = tft.interpret_output(raw_predictions, reduction="sum")

figs = tft.plot_interpretation(interpretation)

# figs est un dictionnaire {nom: figure}
for nom, fig in figs.items():
    chemin = f"results/tft_attention_{nom}.png"
    fig.savefig(chemin, dpi=120, bbox_inches="tight")
    print("Sauvegardé :", chemin)

plt.show()

print("Terminé.")