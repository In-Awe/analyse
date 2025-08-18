#!/usr/bin/env python3
import typer
from rich import print
from backend.data.loader import load_csv, df_to_context

app = typer.Typer()

@app.command()
def ingest(path: str):
    df = load_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    print(df.tail(3).to_string())

@app.command()
def analyze(path: str, lookback: int = 60):
    df = load_csv(path)
    recs = df_to_context(df, lookback)
    print("Context sample:")
    for r in recs[:3]:
        print(r)

if __name__ == '__main__':
    app()
