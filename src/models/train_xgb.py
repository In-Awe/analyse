#!/usr/bin/env python3
"""
Train an XGBoost classifier on the selected features.

Saves:
 - model (joblib) to artifacts/models/xgb
 - metrics JSON to artifacts/models/xgb/xgb_metrics.json

Uses early stopping on validation set.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def load_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def prepare_data(df: pd.DataFrame, target_col: str):
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col].astype(int)
    # map labels to 0..K-1 for XGBoost
    unique_labels = sorted(list(pd.unique(y)))
    label_to_enc = {lab: i for i, lab in enumerate(unique_labels)}
    enc_to_label = {v: k for k, v in label_to_enc.items()}
    y_enc = y.map(label_to_enc).values
    return X, y_enc, label_to_enc, enc_to_label

def metrics(y_true, y_pred, probs=None):
    m = {}
    m["accuracy"] = accuracy_score(y_true, y_pred)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="features parquet with target column")
    parser.add_argument("--target", default="next_5m_dir", help="target column name")
    parser.add_argument("--out-dir", default="artifacts/models/xgb", help="output dir")
    parser.add_argument("--nrounds", type=int, default=200)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    args = parser.parse_args(argv)

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_features(args.features)
    if args.target not in df.columns:
        for c in df.columns:
            if "next_" in c and c.endswith("dir"):
                args.target = c
                break
    X, y_enc, label_to_enc, enc_to_label = prepare_data(df, args.target)
    n = len(X)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X.iloc[:train_end], y_enc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y_enc[train_end:val_end]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    num_class = len(np.unique(y_enc))
    params = {
        "objective": "multi:softprob" if num_class>2 else "binary:logistic",
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 0,
    }
    evallist = [(dtrain, "train"), (dval, "eval")]
    bst = xgb.train(params, dtrain, num_boost_round=args.nrounds, evals=evallist, early_stopping_rounds=args.early_stopping_rounds)
    preds_prob = bst.predict(dval)
    if preds_prob.ndim == 2:
        preds_enc = np.argmax(preds_prob, axis=1)
    else:
        # binary returns probability for positive class
        preds_enc = (preds_prob > 0.5).astype(int)
    # decode predictions to original labels for metrics
    preds = np.array([enc_to_label[int(p)] for p in preds_enc])
    y_val_decoded = np.array([enc_to_label[int(p)] for p in y_val])
    m = metrics(y_val_decoded, preds, preds_prob if preds_prob.ndim==2 else None)
    # save bst and feature names
    bst.save_model(str(outdir / "xgb_model.json"))
    # store metadata and label mapping
    meta = {
        "feature_names": X.columns.tolist(),
        "best_ntree_limit": getattr(bst, "best_ntree_limit", None),
        "label_to_enc": {int(k): int(v) for k, v in label_to_enc.items()},
        "enc_to_label": {int(k): int(v) for k, v in enc_to_label.items()},
    }
    with open(outdir / "xgb_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(outdir / "xgb_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"[xgb] saved model to {outdir} metrics={m}")

if __name__ == "__main__":
    main()
