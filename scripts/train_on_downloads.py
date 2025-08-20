# scripts/train_on_downloads.py
import os
import glob
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ARTIFACTS_DIR = "artifacts/models"
DATA_DIR = "data/binance"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ----- Feature Engineering -----
def build_features(df: pd.DataFrame):
    df["return"] = df["close"].pct_change().fillna(0)
    df["rolling_mean"] = df["close"].rolling(20).mean().fillna(method="bfill")
    df["volatility"] = df["return"].rolling(20).std().fillna(0)
    df["target"] = (df["return"].shift(-1) > 0).astype(int)  # predict up/down
    return df.dropna()


# ----- PyTorch LSTM Model -----
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def train_lstm(X_train, y_train, X_test, y_test, pair, epochs=5):
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = LSTMClassifier(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1)
        acc = accuracy_score(y_test_t, preds)

    model_path = os.path.join(ARTIFACTS_DIR, pair.replace("/", "_") + "_lstm.pt")
    torch.save(model.state_dict(), model_path)

    return acc


def train_xgboost(X_train, y_train, X_test, y_test, pair):
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_path = os.path.join(ARTIFACTS_DIR, pair.replace("/", "_") + "_xgb.json")
    clf.save_model(model_path)

    return acc


def train_on_csv(path):
    pair = os.path.basename(path).replace(".csv", "")
    print(f"Training on {pair}...")
    df = pd.read_csv(path)
    df = build_features(df)

    features = ["close", "volume", "return", "rolling_mean", "volatility"]
    X = df[features].values
    y = df["target"].values

    split = int(len(df) * 0.8)
    X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]

    metrics = {}
    metrics["xgboost_acc"] = train_xgboost(X_train, y_train, X_test, y_test, pair)
    metrics["lstm_acc"] = train_lstm(X_train, y_train, X_test, y_test, pair)

    with open(os.path.join(ARTIFACTS_DIR, pair.replace("/", "_") + "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Finished {pair} → {metrics}")


def main():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("No downloaded Binance CSVs found.")
        return

    for path in csv_files:
        train_on_csv(path)


if __name__ == "__main__":
    main()
