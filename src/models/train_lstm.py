#!/usr/bin/env python3
"""
Train a simple LSTM classifier (PyTorch) on sequences of features.

Assumptions:
 - features parquet is a dense table with timestamp; we'll construct sliding windows (seq_len)
 - for speed and determinism, this script uses a small LSTM and trains for few epochs by default.

Saves:
 - model state_dict to artifacts/models/lstm/lstm_model.pt
 - metrics JSON to artifacts/models/lstm/lstm_metrics.json
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.X) - self.seq_len)
    def __getitem__(self, idx):
        x = self.X[idx:idx+self.seq_len]
        y = self.y[idx+self.seq_len]  # predict next step
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out

def load_features(path: str):
    return pd.read_parquet(path)

def build_sequences(df: pd.DataFrame, target_col: str, seq_len: int, drop_cols=None):
    drop_cols = drop_cols or ["timestamp"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors='ignore')
    X = X.select_dtypes(include=[np.number]).fillna(0).values
    y = df[target_col].values.astype(int)
    # encode labels
    unique_labels = sorted(list(pd.unique(df[target_col].astype(int))))
    label_to_enc = {lab: i for i, lab in enumerate(unique_labels)}
    enc_to_label = {v: k for k, v in label_to_enc.items()}
    y_enc = np.array([label_to_enc[int(v)] for v in y])
    return X, y_enc, label_to_enc, enc_to_label

def compute_metrics(y_true, y_pred):
    m = {}
    m["accuracy"] = accuracy_score(y_true, y_pred)
    m["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return m

def train_loop(model, dl, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in dl:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dl.dataset)

def eval_loop(model, dl):
    model.eval()
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_trues.extend(yb.numpy().tolist())
            y_preds.extend(preds.tolist())
    return compute_metrics(np.array(y_trues), np.array(y_preds))

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="features parquet with target column")
    parser.add_argument("--target", default="next_5m_dir", help="target column")
    parser.add_argument("--seq-len", type=int, default=60, help="sequence length (minutes)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out-dir", default="artifacts/models/lstm")
    args = parser.parse_args(argv)

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_features(args.features)
    if args.target not in df.columns:
        for c in df.columns:
            if "next_" in c and c.endswith("dir"):
                args.target = c
                break
    X, y, label_to_enc, enc_to_label = build_sequences(df, args.target, args.seq_len)
    # simple time-split
    n = len(X)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:train_end])
    X_val = scaler.transform(X[train_end:val_end])
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    train_ds = SequenceDataset(X_train, y_train, args.seq_len)
    val_ds = SequenceDataset(X_val, y_val, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_size = X.shape[1]
    num_classes = len(np.unique(y))
    model = SimpleLSTM(input_size, hidden_size=64, num_layers=1, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        loss = train_loop(model, train_loader, optimizer, criterion)
        val_metrics = eval_loop(model, val_loader)
        print(f"[lstm] epoch {epoch+1}/{args.epochs} loss={loss:.4f} val_f1={val_metrics['f1_macro']:.4f}")
    # final eval on validation set
    final_metrics = eval_loop(model, val_loader)
    # save model + scaler + metrics
    torch.save(model.state_dict(), outdir / "lstm_model.pt")
    # save scaler via joblib
    import joblib
    joblib.dump(scaler, outdir / "lstm_scaler.joblib")
    # save label mapping
    import json
    with open(outdir / "label_mapping.json", "w") as f:
        label_to_enc_json = {int(k): int(v) for k, v in label_to_enc.items()}
        enc_to_label_json = {int(k): int(v) for k, v in enc_to_label.items()}
        json.dump({"label_to_enc": label_to_enc_json, "enc_to_label": enc_to_label_json}, f, indent=2)
    with open(outdir / "lstm_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"[lstm] saved model to {outdir} metrics={final_metrics}")

if __name__ == "__main__":
    main()
