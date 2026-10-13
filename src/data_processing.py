"""Chargement, nettoyage et splits temporels."""

import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def clean_data(data, save_path=None):
    cleaned_data = data.copy()

    if "Date" in cleaned_data.columns:
        cleaned_data["Date"] = pd.to_datetime(cleaned_data["Date"])
        cleaned_data = cleaned_data.sort_values("Date").reset_index(drop=True)
        cleaned_data = cleaned_data.drop_duplicates(subset="Date", keep="last")

    if save_path is not None :
        cleaned_data.to_csv(save_path,index = False)
    return cleaned_data


def split_data(data, test_size=0.2):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = max(1, int(len(data) * (1 - test_size)))
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()
    return train_data, test_data


if __name__ == "__main__":
    data = load_data("data/AAPL.csv")
    cleaned_data = clean_data(data,save_path ="data/AAPl_clean.csv")
    train_data, test_data = split_data(cleaned_data, test_size=0.2)
    print("Train :", train_data.shape)
    print("Test :", test_data.shape)
    print(train_data.head())
    print("Données d'entraînement :")
    print(train_data.head())
    print("\nDonnées de test :")
    print(test_data.head())