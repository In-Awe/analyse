# Binance download & training orchestration

Files:
 - tools/binance_public_fetcher.py  : public REST klines fetcher
 - scripts/download_all_pairs.py    : orchestrates downloads for multiple pairs
 - scripts/train_on_downloads.py    : simulates training on downloaded CSVs
 - ui/streamlit_app.py              : local UI to start downloads and training

Security:
 - No API key or secret is required for public historical klines.
 - Optional API key may be provided for rate limit improvements (not stored in files).

Usage examples:
 - python scripts/download_all_pairs.py --symbols BTCUSDT,ETHUSDT --months 84
 - python scripts/train_on_downloads.py --raw artifacts/raw --limit 10
 - streamlit run ui/streamlit_app.py  # run UI locally

How to run (quick start)

Ensure your virtualenv has requests, pandas, streamlit, dill:

pip install requests pandas streamlit dill


Ensure gating is set if you want to allow operations. The scripts check configs/global.yaml — training is gated, downloads are gated by the start wrapper too.

From repo root, to download a few pairs (local test):

python scripts/download_all_pairs.py --symbols BTCUSDT,ETHUSDT --months 24 --out artifacts/raw


To simulate training on downloaded data:

python scripts/train_on_downloads.py --raw artifacts/raw --limit 5


To run the UI locally:

streamlit run ui/streamlit_app.py
# open browser to localhost:8501
