#!/usr/bin/env python3
"""
scripts/upload_artifacts_s3.py

Uploads directories (raw downloads, models, training) to S3 for long-term storage.
Also writes a local manifest of raw files for reproducibility.

Usage:
  python scripts/upload_artifacts_s3.py --src artifacts/raw --bucket my-bucket --prefix myproject/raw-2025-08 --upload-manifest
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime

def ensure_boto():
    try:
        import boto3  # type: ignore
        return boto3
    except Exception as e:
        print("boto3 is required for uploading to S3. Install with: pip install boto3")
        raise

def collect_manifest(local_dir: Path):
    manifest = []
    for root, _, files in os.walk(local_dir):
        for fname in files:
            lp = Path(root) / fname
            stat = lp.stat()
            manifest.append({
                "path": str(lp.relative_to(local_dir)),
                "absolute_path": str(lp.resolve()),
                "size": stat.st_size,
                "mtime": stat.st_mtime
            })
    manifest.sort(key=lambda x: x["path"])
    return manifest

def upload_dir_to_s3(local_dir: Path, bucket: str, prefix: str, boto3_mod):
    s3 = boto3_mod.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=os.environ.get("AWS_REGION")
    )
    if not local_dir.exists():
        print(f"Local directory {local_dir} not found. Skipping.")
        return
    local_dir = local_dir.resolve()
    for root, _, files in os.walk(local_dir):
        for fname in files:
            lp = Path(root) / fname
            rel = lp.relative_to(local_dir)
            s3_key = f"{prefix.rstrip('/')}/{rel.as_posix()}" if prefix else rel.as_posix()
            print(f"Uploading {lp} -> s3://{bucket}/{s3_key}")
            s3.upload_file(str(lp), bucket, s3_key)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="artifacts/raw", help="Local dir to upload")
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="", help="S3 key prefix")
    p.add_argument("--upload-manifest", action="store_true", help="Write and upload a manifest of the src dir")
    args = p.parse_args()
    boto3 = ensure_boto()
    local_dir = Path(args.src)
    if args.upload_manifest:
        manifest = collect_manifest(local_dir)
        manifest_path = Path("artifacts") / "raw_manifest.json"
        manifest_obj = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "src": str(local_dir),
            "file_count": len(manifest),
            "files": manifest
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")
        print(f"Wrote local manifest to {manifest_path}")
    upload_dir_to_s3(local_dir, args.bucket, args.prefix or "", boto3)
    if args.upload_manifest:
        # upload manifest as well
        s3 = boto3.client("s3")
        s3_key = f"{args.prefix.rstrip('/')}/raw_manifest.json" if args.prefix else "raw_manifest.json"
        s3.upload_file(str(manifest_path), args.bucket, s3_key)
        print(f"Uploaded manifest to s3://{args.bucket}/{s3_key}")
    print("Upload complete.")

if __name__ == "__main__":
    main()
