#!/usr/bin/env python3
"""
scripts/model_registry.py

A minimal SQLite-backed model registry.

Tables:
 - models: id (int), model_file (text), model_type, pair, git_sha, run_id, created_at, metrics_json (text)

This is intentionally small and local; you can replace with an external registry later.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import json
from datetime import datetime

DB_PATH = Path("artifacts/registry/models.db")
DB_DIR = DB_PATH.parent

def init_db():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_file TEXT NOT NULL,
        model_type TEXT,
        pair TEXT,
        git_sha TEXT,
        run_id TEXT,
        created_at TEXT,
        metrics_json TEXT
    )""")
    conn.commit()
    conn.close()

def register_model(model_file: str, model_type: str, pair: str, git_sha: str, run_id: str, metrics: dict):
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO models (model_file, model_type, pair, git_sha, run_id, created_at, metrics_json)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (model_file, model_type, pair, git_sha, run_id, datetime.utcnow().isoformat()+"Z", json.dumps(metrics)))
    conn.commit()
    conn.close()
    print(f"Registered model in registry: {model_file}")

def query_models(limit: int = 100):
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT id, model_file, model_type, pair, git_sha, run_id, created_at FROM models ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    # simple CLI to list
    rows = query_models(50)
    for r in rows:
        print(r)
