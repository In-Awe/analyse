#!/usr/bin/env python3
"""
Train a logistic regression baseline (one-vs-rest for multi-class).

Saves:
 - model artifact (joblib) to artifacts/models/lr_model.joblib
 - metrics CSV to artifacts/models/lr_metrics.json
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def load_features(path: str):
    return pd.read_parquet(path)

def prepare_Xy(df: pd.DataFrame, target_col: str, drop_cols=None):
    drop_cols = drop_cols or ["timestamp"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors='ignore')
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col].astype(int)
    # encode labels to 0..K-1 for consistency across models
    unique_labels = sorted(list(pd.unique(y)))
    label_to_enc = {lab: i for i, lab in enumerate(unique_labels)}
    enc_to_label = {v: k for k, v in label_to_enc.items()}
    y_enc = y.map(label_to_enc)
    return X, y_enc, label_to_enc, enc_to_label

def train_lr(X_train, y_train, X_val, y_val, seed=42):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xval_s = scaler.transform(X_val)
    clf = LogisticRegression(max_iter=200, multi_class="ovr", solver="lbfgs", random_state=seed)
    clf.fit(Xtr_s, y_train)
    preds = clf.predict(Xval_s)
    probs = clf.predict_proba(Xval_s)
    return clf, scaler, preds, probs

def metrics(y_true, y_pred, probs=None):
    m = {}
    m["accuracy"] = accuracy_score(y_true, y_pred)
    # For multiclass precision/recall use macro
    m["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        if probs is not None and probs.shape[1] == 2:
            m["auc"] = roc_auc_score(y_true, probs[:,1])
    except Exception:
        m["auc"] = None
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return m

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="features parquet with target column")
    p.add_argument("--target", default="next_5m_dir", help="target column name (e.g., next_5m_dir)")
    p.add_argument("--out-dir", default="artifacts/models/lr", help="output dir")
    args = p.parse_args(argv)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_features(args.features)
    # ensure correct default target name
    if args.target not in df.columns:
        # try common name produced by target.py (match horizon)
        for c in df.columns:
            if "next_" in c and c.endswith("dir"):
                args.target = c
                break
    X, y_enc, label_to_enc, enc_to_label = prepare_Xy(df, args.target)
    # simple time split: first 70% train, next 15% val
    n = len(X)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X.iloc[:train_end], y_enc.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y_enc.iloc[train_end:val_end]

    clf, scaler, preds_enc, probs = train_lr(X_train, y_train, X_val, y_val)
    # decode predictions back to original labels
    preds = np.array([enc_to_label[int(p)] for p in preds_enc])
    y_val_decoded = np.array([enc_to_label[int(p)] for p in y_val])
    m = metrics(y_val_decoded, preds, probs if probs is not None else None)
    # save artifacts
    joblib.dump({"model": clf, "scaler": scaler, "label_to_enc": label_to_enc, "enc_to_label": enc_to_label}, outdir / "lr_artifact.joblib")
    with open(outdir / "lr_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"[lr] saved model to {outdir} metrics={m}")

if __name__ == "__main__":
    main()
