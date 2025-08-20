#!/usr/bin/env python3
"""
tools/db_ingest.py

Simple CSV -> Postgres ingestion using SQLAlchemy. Meant for local dev storage of candles.
Requires: sqlalchemy, psycopg2-binary, pandas

Usage:
  pip install sqlalchemy psycopg2-binary pandas
  docker-compose up -d
  python tools/db_ingest.py artifacts/batch/BTC_2025-07/cleaned.csv
"""
import os
import sys
import pandas as pd
from sqlalchemy import create_engine

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://analyse:analysepass@localhost:5432/analyse_db")

def write_candles(csv_path, table_name="candles"):
    df = pd.read_csv(csv_path, parse_dates=['open_time','close_time'], infer_datetime_format=True)
    engine = create_engine(DB_URL)
    # write in chunks to avoid huge memory spikes
    df.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=1000)
    print("Wrote", len(df), "rows to", table_name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/db_ingest.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    write_candles(csv_path)
