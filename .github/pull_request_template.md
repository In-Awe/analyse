## Summary

This PR introduces a Python backend CLI with supporting modules for analysis.

## Features
- Ingest Binance CSVs
- Compute technical indicators
- Run local LLM inference (quantised when possible)
- Run a toy LoRA training job
- Fetch historical klines from Binance
- Run a naive backtest

## Setup (macOS Quickstart)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
