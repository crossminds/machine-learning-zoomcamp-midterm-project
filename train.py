"""Train the final fraud detection model and save it to models/model.bin.

Reproduces the final steps of notebook.ipynb with the hyperparameters selected
there (XGBoost, see notebook sections 4-6):

1. load and clean the data (drop duplicates), split 60/20/20 as in the notebook
2. pick the decision threshold that maximizes F1 on 5-fold out-of-fold
   predictions over train + validation
3. train the final model on train + validation
4. evaluate once on the held-out test set
5. save {pipeline, threshold, metadata} with pickle

Usage:
    uv run python train.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from common import RAW_COLUMNS, SEED, TARGET, add_features, clean_data, load_data, split_data

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.bin"

# Selected in notebook.ipynb (section 4.3, confirmed by 5-fold CV in section 5)
XGB_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "min_child_weight": 1,
    "scale_pos_weight": 10.0,
    "n_estimators": 165,
}


def make_pipeline() -> Pipeline:
    """Raw transaction columns in -> fraud probability out."""
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("model", xgb.XGBClassifier(n_jobs=-1, random_state=SEED, **XGB_PARAMS)),
    ])


def find_threshold(X: pd.DataFrame, y: np.ndarray) -> float:
    """F1-optimal threshold on 5-fold out-of-fold predictions."""
    oof = np.zeros(len(y))
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(X, y)):
        model = make_pipeline().fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        print(f"  fold {fold}: PR-AUC = {average_precision_score(y[va_idx], oof[va_idx]):.4f}")

    precision, recall, thresholds = precision_recall_curve(y, oof)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))
    print(f"  CV PR-AUC (out-of-fold) = {average_precision_score(y, oof):.4f}")
    print(f"  threshold = {thresholds[best]:.4f} "
          f"(precision {precision[best]:.3f}, recall {recall[best]:.3f}, F1 {f1[best]:.3f})")
    return float(thresholds[best])


def main():
    print("Loading data ...")
    df = clean_data(load_data())
    df_train, df_val, df_test = split_data(df)
    df_full_train = pd.concat([df_train, df_val], ignore_index=True)
    X_full, y_full = df_full_train[RAW_COLUMNS], df_full_train[TARGET].values
    X_test, y_test = df_test[RAW_COLUMNS], df_test[TARGET].values
    print(f"  train+val: {len(df_full_train)} rows ({y_full.sum()} fraud), "
          f"test: {len(df_test)} rows ({y_test.sum()} fraud)")

    print("Choosing the decision threshold with 5-fold CV ...")
    threshold = find_threshold(X_full, y_full)

    print("Training the final model on train+val ...")
    pipeline = make_pipeline().fit(X_full, y_full)

    y_pred = pipeline.predict_proba(X_test)[:, 1]
    y_flag = (y_pred >= threshold).astype(int)
    metrics = {
        "test_pr_auc": average_precision_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_flag),
        "test_recall": recall_score(y_test, y_flag),
    }
    print("Test set:", {k: round(v, 4) for k, v in metrics.items()})

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "model_name": "xgboost",
        "params": XGB_PARAMS,
        "metrics": metrics,
    }
    with open(MODEL_PATH, "wb") as f_out:
        pickle.dump(artifact, f_out)
    print(f"Saved model to {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
