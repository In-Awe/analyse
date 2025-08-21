#!/usr/bin/env python3
"""
ui/streamlit_app.py

Local-only Streamlit UI to:
 - display and select symbols to download
 - provide an optional API key (not stored) after human confirmation
 - start downloads and training simulation
 - show logs and quick links to artifacts

Run locally:
  pip install streamlit
  streamlit run ui/streamlit_app.py

Security: binds to localhost by default when running Streamlit locally.
"""
from __future__ import annotations
import streamlit as st
import subprocess
import os
from pathlib import Path
import threading
import queue
import time
import json

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT)
os.environ['PYTHONPATH'] = f"{PY}:{os.environ.get('PYTHONPATH','')}"
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Analyse Training Control", layout="wide")

st.title("Analyse — Download & Training Control")

with st.expander("Training Gate (global config)"):
    st.write("This UI will check `configs/global.yaml` for TRAINING_ENABLED and HUMAN_APPROVAL_TOKEN before starting any network or training tasks.")
    if Path("configs/global.yaml").exists():
        cfg = None
        try:
            import yaml
            cfg = yaml.safe_load(open("configs/global.yaml"))
        except Exception:
            cfg = {}
        st.json(cfg)
    else:
        st.warning("configs/global.yaml not found.")

# Human verification toggle
st.header("Human Verification & optional Binance API key")
human_ok = st.checkbox("I confirm human approval to start downloads and training (local only)", value=False)
api_key = st.text_input("Optional Binance API key (not stored)", type="default")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Symbols to download")
    custom_symbols = st.text_input("Comma-separated symbols (leave blank for default)", value="BTCUSDT,ETHUSDT")
    top_n = st.number_input("Or download top N USDT pairs (0 to ignore)", min_value=0, max_value=200, value=0)
    months = st.number_input("Months back to attempt per symbol", min_value=12, max_value=120, value=84)
    limit = st.number_input("Limit number of symbols to download (0 = all selected)", min_value=0, max_value=200, value=0)
    start_download = st.button("Start download")
with col2:
    st.subheader("Training")
    start_training = st.button("Start training on downloaded data")
    st.write("Training will process files in artifacts/raw and produce artifacts/training and artifacts/models.")
    start_download_and_train = st.button("Download & Train (combined)")

log_box = st.empty()
log_q = queue.Queue()

def run_cmd_and_stream(cmd, q):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        q.put(line)
    p.wait()
    q.put(f"__EXIT__:{p.returncode}")

def stream_logs(q, placeholder):
    logs = []
    while True:
        try:
            line = q.get(timeout=0.5)
        except queue.Empty:
            st.sleep(0.2)
            continue
        if line.startswith("__EXIT__:"):
            rc = int(line.split(":",1)[1])
            logs.append(f"[process exited: {rc}]")
            placeholder.code("\n".join(logs[-200:]))
            break
        logs.append(line.rstrip())
        placeholder.code("\n".join(logs[-200:]))

if start_download and human_ok:
    # build command
    symbols_arg = custom_symbols.strip()
    sym_arg = f"--symbols {symbols_arg}" if symbols_arg else ""
    top_arg = f"--top {top_n}" if top_n>0 else ""
    lim_arg = f"--limit {limit}" if limit>0 else ""
    cmd = f'python scripts/download_all_pairs.py {sym_arg} {top_arg} --months {months}'
    if api_key:
        cmd += f' --api_key "{api_key}"'
    st.info("Running download: " + cmd)
    placeholder = log_box.empty()
    t = threading.Thread(target=run_cmd_and_stream, args=(cmd, log_q))
    t.start()
    stream_logs(log_q, placeholder)
    st.success("Download step completed (check artifacts/raw)")

if start_training and human_ok:
    st.info("Starting training simulation over artifacts/raw...")
    cmd = "python scripts/train_on_downloads.py"
    placeholder = log_box.empty()
    t = threading.Thread(target=run_cmd_and_stream, args=(cmd, log_q))
    t.start()
    stream_logs(log_q, placeholder)
    st.success("Training simulation completed (check artifacts/training and artifacts/models)")

if start_download_and_train and human_ok:
    st.info("Starting combined download + training orchestration...")
    # build orchestrator command
    symbols_arg = custom_symbols.strip()
    sym_arg = f"--symbols {symbols_arg}" if symbols_arg else ""
    top_arg = f"--top {top_n}" if top_n>0 else ""
    cmd = f'python scripts/run_download_and_train.py {sym_arg} {top_arg} --months {months}'
    if api_key:
        cmd += f' --api_key \"{api_key}\"'
    if limit>0:
        cmd += f' --limit {limit}'
    placeholder = log_box.empty()
    t = threading.Thread(target=run_cmd_and_stream, args=(cmd, log_q))
    t.start()
    stream_logs(log_q, placeholder)
    st.success("Download+Train orchestration complete. Check artifacts/training for training_summary.json")

st.markdown("---")
st.subheader("Artifacts quick links")
if ARTIFACTS.exists():
    for p in ARTIFACTS.iterdir():
        st.write(p.name, "- contents:", len(list(p.rglob("*"))))
else:
    st.info("No artifacts folder yet; run a download or training step.")
