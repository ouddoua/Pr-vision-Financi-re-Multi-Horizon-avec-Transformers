import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import sys
sys.path.append("src")

from models import load_and_prepare, LSTMModel


parser = argparse.ArgumentParser(description="Entraîne le LSTM multi-horizon.")
parser.add_argument("--output", default="data/lstm_model.pt")
args = parser.parse_args()

# 1. Charger les données
X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare()
print("Train :", X_train.shape, "| Val :", X_val.shape)

# 2. Convertir en tenseurs
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# 3. Créer le modèle
model = LSTMModel(n_features=X_train.shape[2])
# 4. Perte + optimiseur
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Entraîner avec early stopping
meilleur_val = 999
meilleur_etat = None
patience = 10
compteur = 0

for epoch in range(100):
    # Train
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    optimizer.step()

    # Val
    model.eval()
    with torch.no_grad():
        pred_val = model(X_val)
        loss_val = loss_fn(pred_val, y_val)

    print(f"Epoch {epoch+1:2d} | train = {loss.item():.6f} | val = {loss_val.item():.6f}")

    # Sauvegarder le meilleur
    if loss_val.item() < meilleur_val:
        meilleur_val = loss_val.item()
        meilleur_etat = copy.deepcopy(model.state_dict())
        compteur = 0
    else:
        compteur += 1
        if compteur >= patience:
            print(f"\nArrêt à l'époque {epoch+1} — pas d'amélioration depuis {patience} époques.")
            break

# 6. Recharger le meilleur modèle et sauvegarder
model.load_state_dict(meilleur_etat)
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), output_path)
print(f"\nMeilleure val_loss : {meilleur_val:.6f}")
print(f"Modèle sauvegardé : {output_path}")