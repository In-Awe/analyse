#!/usr/bin/env python3
"""
scripts/generate_training_report.py

Collects training artifacts (training_summary.json, per-model metrics, progress telemetry,
and recent logs) and emits:
 - artifacts/reports/<runid>_report.json
 - artifacts/reports/<runid>_report.md

Usage:
  python scripts/generate_training_report.py --summary artifacts/training/training_summary.json --out artifacts/reports

The tool is safe to run multiple times; it will overwrite previous report files for the same run_id.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import glob
import datetime
import textwrap

def find_latest_progress(summary_dir: Path):
    files = sorted(summary_dir.glob("progress_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def load_json_if_exists(p: Path):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": str(e)}
    return None

def collect_model_metrics(models_dir: Path):
    metrics_files = sorted(models_dir.glob("*_metrics.json"))
    out = {}
    for mf in metrics_files:
        try:
            j = json.loads(mf.read_text(encoding="utf-8"))
            out[mf.name] = j
        except Exception as e:
            out[mf.name] = {"_error": str(e)}
    return out

def tail_file_lines(p: Path, n=200):
    if not p.exists():
        return []
    with p.open("rb") as f:
        # naive tail implementation:
        try:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b''
            while size > 0 and data.count(b'\n') <= n:
                read_size = min(block, size)
                f.seek(size - read_size)
                chunk = f.read(read_size)
                data = chunk + data
                size -= read_size
            lines = data.splitlines()[-n:]
            return [l.decode(errors='replace') for l in lines]
        except Exception:
            f.seek(0)
            return [l.decode(errors='replace') for l in f.readlines()[-n:]]

def generate_markdown(report: dict, out_md: Path):
    lines = []
    lines.append(f"# Training report — run {report.get('run_id','unknown')}")
    lines.append(f"*Git SHA:* `{report.get('git_sha','unknown')}`")
    lines.append(f"*Run start:* {report.get('run_start_unix')}")
    lines.append(f"*Run end:* {report.get('run_end_unix')}")
    lines.append(f"*Duration (s):* {report.get('duration_seconds')}")
    lines.append(f"*Models trained:* {report.get('models_trained')}")
    lines.append("")
    lines.append("## Details")
    for d in report.get("details", []):
        lines.append(f"### {d.get('pair')} — {d.get('model_type')}")
        lines.append(f"- model_path: `{d.get('model_path')}`")
        lines.append("- metrics:")
        metrics = d.get("metrics", {})
        if isinstance(metrics, dict):
            for k,v in metrics.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append(f"  - {metrics}")
        lines.append("")
    lines.append("## Per-model metrics (collected files)")
    for mname, mm in report.get("model_metrics", {}).items():
        lines.append(f"### {mname}")
        if isinstance(mm, dict):
            for k,v in mm.items():
                lines.append(f"- {k}: {v}")
        else:
            lines.append(f"- {mm}")
        lines.append("")
    lines.append("## Telemetry (latest progress file)")
    prog = report.get("progress", {})
    lines.append("```json")
    lines.append(json.dumps(prog, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Tail of recent logs (if available)")
    logs = report.get("recent_logs", [])
    lines.append("```\n" + "\n".join(logs) + "\n```")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="artifacts/training/training_summary.json", help="Path to training_summary.json")
    p.add_argument("--models", default="artifacts/models", help="Models dir where per-model metrics are stored")
    p.add_argument("--progress_dir", default="artifacts/training", help="Progress/telemetry directory")
    p.add_argument("--logs", default="logs", help="Path to logs directory (optional)")
    p.add_argument("--out", default="artifacts/reports", help="Output directory for report")
    p.add_argument("--tail", type=int, default=200, help="Number of log lines to include")
    args = p.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print("Summary not found at", summary_path)
        return 2

    summary = load_json_if_exists(summary_path)
    model_metrics = collect_model_metrics(Path(args.models))
    progress_file = find_latest_progress(Path(args.progress_dir))
    progress = load_json_if_exists(progress_file) if progress_file else {}

    # collect recent logs
    recent_logs = []
    logs_dir = Path(args.logs)
    if logs_dir.exists():
        # look for most recently modified file
        cand = sorted(logs_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cand:
            recent_logs = tail_file_lines(cand[0], n=args.tail)
    else:
        # also try to find any training log under artifacts (grep)
        artlogs = list(Path("artifacts").rglob("*.log"))
        if artlogs:
            recent_logs = tail_file_lines(sorted(artlogs, key=lambda p: p.stat().st_mtime, reverse=True)[0], n=args.tail)

    # build final report object
    run_id = summary.get("run_id", datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = out_dir / f"{run_id}_report.json"
    report_md_path = out_dir / f"{run_id}_report.md"

    report = {
        "run_id": run_id,
        "git_sha": summary.get("git_sha"),
        "run_start_unix": summary.get("run_start_unix"),
        "run_end_unix": summary.get("run_end_unix"),
        "duration_seconds": summary.get("duration_seconds"),
        "models_trained": summary.get("models_trained"),
        "details": summary.get("details"),
        "model_metrics": model_metrics,
        "progress": progress,
        "recent_logs": recent_logs
    }

    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    generate_markdown(report, report_md_path)

    print("Wrote report JSON to", report_json_path)
    print("Wrote report MD to", report_md_path)
    print("To print: `pandoc {md} -o {pdf}` or open the md in your editor.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
