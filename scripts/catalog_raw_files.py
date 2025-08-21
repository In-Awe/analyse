import os
import re
import csv

RAW_DIR = "data/raw"
MANIFEST = os.path.join(RAW_DIR, "manifest.csv")

FILENAME_RE = re.compile(r"(?P<symbol>[A-Z]+)USDT_1m_(?P<year>\d{4})-(?P<month>\d{2})\.csv")


def catalog():
    rows = []
    for fname in sorted(os.listdir(RAW_DIR)):
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        symbol = m.group("symbol")
        year = m.group("year")
        month = m.group("month")
        rows.append({
            "filename": fname,
            "symbol": symbol,
            "year": year,
            "month": month
        })

    if not rows:
        print("No valid raw CSVs found in", RAW_DIR)
        return

    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "symbol", "year", "month"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest with {len(rows)} entries to {MANIFEST}")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    catalog()
