"""
ui/training_panel.py

Small Streamlit panel to display latest training summary and allow quick replay trigger.
Import this module into your main Streamlit app to embed the panel.

Requires the report/summary files to be present in artifacts/training and artifacts/models.
"""
from __future__ import annotations
import streamlit as st
from pathlib import Path
import json
import subprocess
import threading
import os
import sys

SUMMARY_PATH = Path("artifacts/training/training_summary.json")
MODELS_DIR = Path("artifacts/models")
REPLAY_SCRIPT = Path("scripts/replay_historical.py")

def load_summary():
    if not SUMMARY_PATH.exists():
        return None
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        st.error("Failed to load training_summary.json: " + str(e))
        return None

def run_replay_async(pair: str, from_month: str, to_month: str):
    cmd = [sys.executable, str(REPLAY_SCRIPT), "--pair", pair, "--from", from_month, "--to", to_month, "--run-train"]
    env = os.environ.copy()
    env["PYTHONPATH"] = f'{os.getcwd()}:{env.get("PYTHONPATH","")}'
    def target():
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            st.write(line)
        proc.wait()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread

def panel():
    st.header("Training summary & replay")
    summary = load_summary()
    if not summary:
        st.info("No training_summary.json found (run training first).")
        return
    st.subheader("Latest run")
    st.write(f"Run ID: {summary.get('run_id')}")
    st.write(f"Git SHA: {summary.get('git_sha')}")
    st.write(f"Duration (s): {summary.get('duration_seconds')}")
    st.write(f"Models trained: {summary.get('models_trained')}")
    for d in summary.get("details", []):
        st.markdown(f"**{d.get('pair')}** — {d.get('model_type')}")
        st.write(d.get("metrics", {}))
    st.subheader("Replay historical data for a pair (quick)")
    with st.form("replay-form"):
        pair = st.text_input("Pair (e.g. BTCUSDT)", value="")
        from_month = st.text_input("From month (YYYY-MM)", value="")
        to_month = st.text_input("To month (YYYY-MM)", value="")
        submitted = st.form_submit_button("Start replay & train")
        if submitted:
            if not pair or not from_month or not to_month:
                st.error("Please provide pair, from and to months")
            else:
                st.info("Starting replay in background — logs will appear below.")
                run_replay_async(pair, from_month, to_month)
