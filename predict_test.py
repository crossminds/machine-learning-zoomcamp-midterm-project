"""Send real test-set transactions to the running web service and print the results.

The transactions come from the held-out test split (same split as train.py),
so the model has never seen them.

Usage (with the service running, see README):
    uv run python predict_test.py                     # 5 fraud + 5 legit, localhost:9696
    uv run python predict_test.py --n 20              # 20 of each
    uv run python predict_test.py --url http://host:port
"""

import argparse

import pandas as pd
import requests

from common import RAW_COLUMNS, SEED, TARGET, clean_data, load_data, split_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9696", help="service base URL")
    parser.add_argument("--n", type=int, default=5, help="number of fraud and of legit transactions")
    args = parser.parse_args()

    health = requests.get(f"{args.url}/health", timeout=10)
    health.raise_for_status()
    print(f"Service at {args.url}: {health.json()}\n")

    _, _, df_test = split_data(clean_data(load_data()))
    samples = pd.concat([
        df_test[df_test[TARGET] == 1].sample(args.n, random_state=SEED),
        df_test[df_test[TARGET] == 0].sample(args.n, random_state=SEED),
    ])

    correct = 0
    for _, row in samples.iterrows():
        transaction = {col: float(row[col]) for col in RAW_COLUMNS}
        response = requests.post(f"{args.url}/predict", json=transaction, timeout=10)
        response.raise_for_status()
        result = response.json()

        actual = "fraud" if row[TARGET] == 1 else "legit"
        verdict = "FRAUD" if result["is_fraud"] else "legit"
        ok = result["is_fraud"] == (row[TARGET] == 1)
        correct += ok
        print(
            f"Amount {transaction['Amount']:>9.2f} EUR | "
            f"fraud_probability = {result['fraud_probability']:.4f} | "
            f"predicted: {verdict:5s} | actual: {actual:5s} | {'correct' if ok else 'WRONG'}"
        )

    print(f"\n{correct}/{len(samples)} correct (threshold = {result['threshold']})")


if __name__ == "__main__":
    main()
