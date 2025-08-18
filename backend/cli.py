#!/usr/bin/env python3
"""
backend/cli.py
Entry point for the CLI using Typer.
"""
import os
import typer
from rich import print

from backend.data.loader import load_csv, df_to_context
from backend.binance_client import BinanceClient
from backend.model.infer import analyse_series
from backend.model.lora_train import train_lora
from backend.data.audit import DataAuditor

app = typer.Typer(help="CLI for LLM trading assistant")


@app.command()
def ingest(path: str):
    """
    Load a CSV and print basic info
    """
    df = load_csv(path)
    print(f"[green]Loaded[/green] {len(df)} rows from {path}")
    print(df.tail(3).to_string())


@app.command()
def analyze(path: str, lookback: int = 60):
    """
    Analyse last N rows using local LLM
    """
    df = load_csv(path)
    context = df_to_context(df, lookback)
    out = analyse_series(context)
    print(out)


@app.command()
def fetch(symbol: str, start: str, end: str, out_csv: str = "fetched.csv"):
    """
    Fetch historical klines from Binance and save to CSV
    """
    client = BinanceClient()
    df = client.get_historical_klines(symbol, start, end)
    df.to_csv(out_csv, index=False)
    print(f"[green]Saved[/green] {len(df)} rows to {out_csv}")


@app.command()
def train(dataset: str = "backend/samples/toy_train.json", output_dir: str = "backend/model/checkpoints"):
    """
    Run a toy LoRA training job for proof of concept
    """
    train_lora(dataset, output_dir)
    print("[green]LoRA training finished (toy run)[/green]")


@app.command()
def audit(path: str, out: str = "backend/reports", freq: str = "1T", impute: str = "ffill"):
    """
    Run data audit (gaps, dupes, spikes, imputation, ADF, ACF/PACF, intraday profile).
    Writes a JSON report to `out`.
    """
    df = load_csv(path)
    auditor = DataAuditor(freq=freq)
    result = auditor.audit(df, impute_method=impute)
    report_path = auditor.export_report(result["report"], out_dir=out)
    print(f"[green]Audit saved to[/green] {report_path}")


if __name__ == "__main__":
    app()
