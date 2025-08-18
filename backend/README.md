Backend for LLM trading assistant
=================================

This folder contains a Python CLI and supporting modules to:

- ingest Binance CSVs
- compute technical indicators
- run local LLM inference (quantised when possible)
- run a toy LoRA training job
- fetch historical klines from Binance
- run a naive backtest

Quickstart (macOS)
------------------

1. Create and activate a virtual environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

2. Set environment variables (use `backend/config.example.env` as a template)

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
```

3. Run CLI examples

```bash
# ingest a CSV and print a summary
python cli.py ingest --path /mnt/data/BTCUSDT_1m_January_2025.csv

# analyse last 60 rows using the local model
python cli.py analyze --path /mnt/data/BTCUSDT_1m_January_2025.csv --lookback 60

# fetch historical data from Binance and save to CSV
python cli.py fetch --symbol BTCUSDT --start 2025-01-01 --end 2025-02-01 --out-csv btc_jan.csv

# run a toy LoRA training job
python cli.py train --dataset backend/samples/toy_train.json --output-dir backend/model/checkpoints
```

Notes
-----

- If you are on Apple Silicon, make sure PyTorch MPS support is installed. Check with:
  `python -c "import torch; print(torch.backends.mps.is_available())"`.
- bitsandbytes can be problematic to install on macOS. If installation fails, use CPU fallback. See `backend/model/infer.py` for graceful fallback logic.
- Do not commit real API keys. Use `backend/config.example.env` and a local `.env` file or environment variables.

Project layout
--------------

```
backend/
├─ README.md
├─ requirements.txt
├─ config.example.env
├─ cli.py
├─ data/
│  ├─ __init__.py
│  ├─ loader.py
│  └─ features.py
├─ binance_client.py
├─ model/
│  ├─ __init__.py
│  ├─ infer.py
│  └─ lora_train.py
├─ backtest/
│  ├─ __init__.py
│  ├─ engine.py
│  └─ metrics.py
└─ tests/
   ├─ test_loader.py
   └─ test_binance_client.py
```

If you want a full patch applied automatically, use the accompanying git patch.
