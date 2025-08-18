#!/usr/bin/env python3
"""
Feature selection utilities:
 - correlation-based pruning
 - XGBoost feature importance ranking
 - output: configs/selected_features.txt (top-k)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

def load_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def correlation_prune(df: pd.DataFrame, thresh: float=0.95) -> List[str]:
    # drop non-numeric & meta cols
    numeric = df.select_dtypes(include=[np.number]).copy()
    # drop timestamp if exists
    if "timestamp" in numeric.columns:
        numeric = numeric.drop(columns=["timestamp"])
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > thresh)]
    keep = [c for c in numeric.columns if c not in to_drop]
    return keep

def xgboost_importance(df: pd.DataFrame, target_col: str, top_k: int=50, random_state: int=42) -> List[str]:
    data = df.copy()
    # drop timestamp if exists
    if "timestamp" in data.columns:
        data = data.drop(columns=["timestamp"])
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = xgb.DMatrix(X_train.fillna(0), label=y_train)
    params = {"objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0}
    model = xgb.train(params, dtrain, num_boost_round=100)
    imp = model.get_score(importance_type="gain")
    # map importances to columns
    scores = []
    for f in X.columns:
        k = f"f{X.columns.get_loc(f)}"
        scores.append((f, imp.get(k, 0.0)))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    selected = [f for f, s in scores[:top_k]]
    return selected

def write_selected(selected: List[str], outpath: str):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        for s in selected:
            f.write(s + "\n")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="input features parquet (with target column)")
    parser.add_argument("--target", required=True, help="target column name in features")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--corr-thresh", type=float, default=0.95)
    parser.add_argument("--out", default="configs/selected_features.txt")
    args = parser.parse_args(argv)

    df = load_features(args.features)
    # first prune by correlation
    keep = correlation_prune(df, thresh=args.corr_thresh)
    df_pruned = df[keep + [args.target]] if args.target in df.columns else df[keep]
    # run XGBoost importance on pruned df (we expect binary target)
    selected = xgboost_importance(df_pruned, target_col=args.target, top_k=args.top_k)
    write_selected(selected, args.out)
    print(f"Wrote {len(selected)} selected features to {args.out}")

if __name__ == "__main__":
    main()
