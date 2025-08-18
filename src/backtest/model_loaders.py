#!/usr/bin/env python3
"""
Model loader helpers for the backtester.
Includes:
 - loading xgboost (JSON) models + meta
 - loading PyTorch LSTM model + scaler + label mapping
 - unified predict interface for both model types
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
            labels_enc = np.argmax(preds, axis=1)
            return labels_enc, preds
        else:
            return np.argmax(preds, axis=1), None
    else:
        # binary prob
        if predict_proba:
            probs = np.vstack([1 - preds, preds]).T
            labels = (preds > 0.5).astype(int)
            return labels, probs
        else:
            return (preds > 0.5).astype(int), None

class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def load_lstm_artifact(model_path: str, scaler_path: Optional[str], label_map_path: Optional[str], device: str = "cpu") -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch not available in environment; install torch to use LSTM support.")
    # load scaler
    scaler = None
    if scaler_path and Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
    label_map = {}
    if label_map_path and Path(label_map_path).exists():
        label_map = json.loads(Path(label_map_path).read_text())
    # infer input_size from scaler if present else None
    input_size = None
    if scaler is not None and hasattr(scaler, "mean_"):
        try:
            input_size = len(scaler.mean_)
        except Exception:
            input_size = None
    # load model
    state = torch.load(model_path, map_location=device)
    # The saved model may be either state_dict or entire model - handle both
    model = None
    if isinstance(state, dict) and "lstm_model" in state:
        # saved wrapped artifact
        # Not expected in current training scripts; fall through
        state_dict = state["lstm_model"]
    elif isinstance(state, dict) and any(k.startswith("lstm.") or k.startswith("fc.") for k in state.keys()):
        # state is a state_dict for SimpleLSTM
        state_dict = state
    else:
        # unknown format: try to treat as state_dict
        state_dict = state
    # if input_size unknown, pick feature count from state keys heuristically? fallback to None
    if input_size is None:
        # cannot infer; require caller to provide input_size via meta
        # but we can attempt to infer num_features from fc weight shape
        if "fc.weight" in state_dict:
            # fc.weight shape: (num_classes, hidden_size)
            pass
    # We'll construct a SimpleLSTM wrapper only when caller provides input_size and num_classes via label_map
    num_classes = len(label_map.get("enc_to_label", {})) if label_map else None
    return {"state_dict": state_dict, "scaler": scaler, "label_map": label_map, "input_size": input_size, "num_classes": num_classes}

def predict_lstm(artifact: Dict[str, Any], feature_df: pd.DataFrame, seq_len: int = 60, batch_size: int = 512, device: str = "cpu") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict labels (encoded) and optionally probabilities using the provided LSTM artifact.
    feature_df: numeric-only DataFrame ordered by timestamp for which we want a prediction at each row (requires seq_len previous rows).
    Returns labels_enc (n,) and probs (n, num_classes) or None.
    """
    if torch is None:
        raise RuntimeError("torch not available; cannot run LSTM predictions.")
    scaler = artifact.get("scaler")
    state_dict = artifact.get("state_dict")
    label_map = artifact.get("label_map", {})
    num_classes = artifact.get("num_classes") or len(label_map.get("enc_to_label", {})) or 3
    input_size = artifact.get("input_size") or feature_df.shape[1]
    # ensure numeric columns only and order preserved
    X = feature_df.select_dtypes(include=[np.number]).fillna(0.0).values
    # scale if scaler present (apply to each row)
    if scaler is not None:
        X = scaler.transform(X)
    # build sliding windows: number of valid predictions = len(X) - seq_len + 1; we will pad at start to keep same length (preds align to last index)
    n = len(X)
    if n < seq_len:
        # not enough data: return zeros
        return np.zeros(n, dtype=int), None
    # create windows
    indices = np.arange(n)
    windows = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1])) if hasattr(np.lib.stride_tricks, "sliding_window_view") else None
    # fallback: build via loops (memory heavy but robust)
    if windows is None:
        seqs = []
        for i in range(seq_len-1, n):
            seqs.append(X[i-seq_len+1:i+1])
        seqs = np.stack(seqs, axis=0)  # shape (m, seq_len, features)
    else:
        # sliding window view returns shape (n-seq_len+1, seq_len, features)
        seqs = windows.reshape((n - seq_len + 1, seq_len, X.shape[1]))
    # create DataLoader-like batching manually to avoid torch dependency on DataLoader
    import torch
    device = torch.device(device)
    model = SimpleLSTM(input_size, hidden_size=64, num_layers=1, num_classes=num_classes).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    preds_enc = []
    probs = []
    m = seqs.shape[0]
    for start in range(0, m, batch_size):
        batch = torch.tensor(seqs[start:start+batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(batch)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            lab = p.argmax(axis=1)
            preds_enc.extend(lab.tolist())
            probs.append(p)
    preds_enc = np.array(preds_enc, dtype=int)
    probs = np.vstack(probs) if probs else None
    # align preds to original length by left-padding with zeros (or label for 'sideways' mapped)
    pad_len = seq_len - 1
    padding = np.zeros(pad_len, dtype=int)
    labels_full = np.concatenate([padding, preds_enc], axis=0)
    if probs is not None:
        # pad probs with uniform low-confidence rows
        pad_probs = np.zeros((pad_len, probs.shape[1]))
        probs_full = np.vstack([pad_probs, probs])
    else:
        probs_full = None
    return labels_full, probs_full
