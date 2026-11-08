"""Download the ULB credit card fraud dataset from OpenML and save it as a gzipped CSV.

Source: OpenML dataset 1597 ("creditcard"), https://www.openml.org/d/1597
Original: Machine Learning Group, Universite Libre de Bruxelles (ULB).

OpenML serves the data as ARFF; this script converts it to a plain CSV
(columns: Time, V1..V28, Amount, Class) and writes dataset/creditcard.csv.gz.
Only the Python standard library is used, so it runs without installing anything.

Usage:
    python scripts/download_data.py
"""

import gzip
import io
import urllib.request
from pathlib import Path

OPENML_FILE_URL = "https://www.openml.org/data/v1/download/1673544"  # dataset 1597
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492

OUTPUT = Path(__file__).resolve().parent.parent / "dataset" / "creditcard.csv.gz"


def arff_to_csv_lines(lines):
    """Yield CSV lines (header first) from an iterable of ARFF text lines."""
    columns = []
    in_data = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if not in_data:
            lower = line.lower()
            if lower.startswith("@attribute"):
                columns.append(line.split()[1])
            elif lower.startswith("@data"):
                in_data = True
                yield ",".join(columns)
            continue
        yield line.replace("'", "")  # Class is quoted in ARFF: '0' / '1'


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {OPENML_FILE_URL} ...")

    rows = fraud = 0
    with urllib.request.urlopen(OPENML_FILE_URL) as resp, \
            gzip.open(OUTPUT, "wt", encoding="utf-8", newline="\n") as out:
        text = io.TextIOWrapper(resp, encoding="utf-8")
        for i, csv_line in enumerate(arff_to_csv_lines(text)):
            out.write(csv_line + "\n")
            if i == 0:
                continue
            rows += 1
            fraud += csv_line.endswith(",1")

    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB): {rows} rows, {fraud} fraud")
    if (rows, fraud) != (EXPECTED_ROWS, EXPECTED_FRAUD):
        raise SystemExit(
            f"Unexpected counts: got {rows} rows / {fraud} fraud, "
            f"expected {EXPECTED_ROWS} / {EXPECTED_FRAUD}"
        )


if __name__ == "__main__":
    main()
