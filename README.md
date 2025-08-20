# Analyse

## Phase 5 – Batch Testing

### Adding Raw Data
Place Binance-style CSVs into `data/raw/`:

```
BTCUSDT_1m_2025-07.csv
ETHUSDT_1m_2025-07.csv
...
```

### Catalog Raw Files
Run:
```
python scripts/catalog_raw_files.py
```
This generates `data/raw/manifest.csv` listing all available (symbol, year-month) pairs.

### Run Batch Smoke Tests
Run:
```
python scripts/run_batch_smoke.py --limit 2
```
to process two files, or `--all` to process all.
