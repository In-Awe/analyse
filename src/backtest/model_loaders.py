#!/usr/bin/env python3
"""
Robust model loaders for backtester.
- XGBoost loader: load model + meta, predict
- LSTM loader: load scaler + label mapping + meta, reconstruct model class, load state_dict, predict

This module purposely avoids importing heavy deps at module import time where possible.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

def load_xgb_model_and_meta(model_path: str, meta_path: Optional[str] = None) -> Tuple[xgb.Booster, Dict[str, Any]]:
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    meta = {}
    if meta_path and Path(meta_path).exists():
        meta = json.loads(Path(meta_path).read_text())
    return bst, meta

def predict_xgb(bst: xgb.Booster, X: pd.DataFrame, meta: Dict[str, Any], predict_proba: bool=True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    feat_names = meta.get("feature_names", X.columns.tolist())
    Xm = X.reindex(columns=feat_names).fillna(0.0)
    dmat = xgb.DMatrix(Xm.values, feature_names=feat_names)
    preds = bst.predict(dmat)
    if preds.ndim == 2:
        if predict_proba:
            return np.argmax(preds, axis=1), preds
        else:
            return np.argmax(preds, axis=1), None
    else:
        if predict_proba:
            probs = np.vstack([1 - preds, preds]).T
            labels = (preds > 0.5).astype(int)
            return labels, probs
        else:
            return (preds > 0.5).astype(int), None

# Lightweight SimpleLSTM skeleton that will be constructed according to meta
class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def load_lstm_artifact(model_path: str, scaler_path: Optional[str], label_map_path: Optional[str], meta_path: Optional[str], device: str = "cpu") -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch not available. Install PyTorch to use LSTM support.")
    scaler = None
    if scaler_path and Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
    label_map = {}
    if label_map_path and Path(label_map_path).exists():
        label_map = json.loads(Path(label_map_path).read_text())
    meta = {}
    if meta_path and Path(meta_path).exists():
        meta = json.loads(Path(meta_path).read_text())
    # load saved object
    obj = torch.load(model_path, map_location=device)
    # determine state_dict
    if isinstance(obj, dict) and any(k.startswith("lstm") or k.startswith("fc") or k.startswith("module.") for k in obj.keys()):
        state_dict = obj
    elif hasattr(obj, "state_dict"):
        state_dict = obj.state_dict()
    else:
        state_dict = obj
    return {"state_dict": state_dict, "scaler": scaler, "label_map": label_map, "meta": meta, "device": device}

def predict_lstm(artifact: Dict[str, Any], feature_df: pd.DataFrame, seq_len: int = 60, batch_size: int = 512, device: str = "cpu") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if torch is None:
        raise RuntimeError("torch not available. Install PyTorch to use LSTM prediction.")
    meta = artifact.get("meta", {})
    state_dict = artifact.get("state_dict")
    scaler = artifact.get("scaler")
    label_map = artifact.get("label_map", {})
    # infer params from meta or defaults
    input_size = int(meta.get("input_size", feature_df.select_dtypes(include=[np.number]).shape[1]))
    num_classes = int(meta.get("num_classes", len(label_map.get("enc_to_label", {})) or 3))
    hidden_size = int(meta.get("hidden_size", 64))
    num_layers = int(meta.get("num_layers", 1))
    dropout = float(meta.get("dropout", 0.1))
    # prepare numeric X
    X = feature_df.select_dtypes(include=[np.number]).fillna(0.0).values
    if scaler is not None:
        X = scaler.transform(X)
    n = len(X)
    if n < seq_len:
        # not enough data to form a single sequence — return zeros
        return np.zeros(n, dtype=int), None
    # build sequences (sliding windows)
    seqs = []
    for i in range(seq_len - 1, n):
        seqs.append(X[i-seq_len+1:i+1])
    seqs = np.stack(seqs, axis=0)  # (m, seq_len, input_size)
    import torch as _torch
    device = _torch.device(device)
    model = SimpleLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout).to(device)
    # try load state_dict safely (handle "module." prefix)
    sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            sd[k[len("module."):]] = v
        else:
            sd[k] = v
    model.load_state_dict(sd)
    model.eval()
    preds_enc = []
    probs = []
    for start in range(0, seqs.shape[0], batch_size):
        batch = _torch.tensor(seqs[start:start+batch_size], dtype=_torch.float32, device=device)
        with _torch.no_grad():
            logits = model(batch)
            p = _torch.softmax(logits, dim=1).cpu().numpy()
            lab = p.argmax(axis=1)
            preds_enc.extend(lab.tolist())
            probs.append(p)
    preds_enc = np.array(preds_enc, dtype=int)
    probs = np.vstack(probs) if len(probs) > 0 else None
    pad_len = seq_len - 1
    labels_full = np.concatenate([np.zeros(pad_len, dtype=int), preds_enc], axis=0)
    if probs is not None:
        pad_probs = np.zeros((pad_len, probs.shape[1]))
        probs_full = np.vstack([pad_probs, probs])
    else:
        probs_full = None
    # decode encoded labels if label_map present
    enc_to_label = label_map.get("enc_to_label", {})
    if enc_to_label and isinstance(enc_to_label, dict):
        decoded = np.array([int(enc_to_label.get(str(int(x)), int(x))) for x in labels_full], dtype=int)
    else:
        decoded = labels_full
    return decoded, probs_full
