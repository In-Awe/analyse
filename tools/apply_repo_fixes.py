#!/usr/bin/env python3
"""
apply_repo_fixes.py

Small helper script to apply repository lightweight fixes that are safe to run in CI or locally.
Currently:
 - Replace occurrences of .fillna(method="ffill") with .ffill() to avoid FutureWarning and upcoming errors.

This script is idempotent and will write files in-place.
"""
import io
import os
import sys
from pathlib import Path

ROOT = Path(".").resolve()
PATTERN = '.fillna(method="ffill")'
REPL = '.ffill()'

def files_to_fix(root):
    for p in root.rglob("*.py"):
        # skip virtualenvs, venv folders, .git
        if "/.venv/" in str(p) or "/venv/" in str(p) or "/.git/" in str(p):
            continue
        yield p

def fix_file(path: Path):
    text = path.read_text(encoding="utf-8")
    if PATTERN in text:
        new_text = text.replace(PATTERN, REPL)
        path.write_text(new_text, encoding="utf-8")
        print(f"Patched: {path}")
        return True
    return False

def main():
    fixed = 0
    for f in files_to_fix(ROOT):
        try:
            if fix_file(f):
                fixed += 1
        except Exception as e:
            print(f"Error patching {f}: {e}", file=sys.stderr)
    print(f"Done. Files patched: {fixed}")

if __name__ == "__main__":
    main()
