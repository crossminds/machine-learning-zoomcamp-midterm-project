# 💳 credit-card-fraud-detection

A machine learning service that scores credit card transactions for fraud in
real time. A payment system sends a transaction to the service and gets back
the probability that it is fraudulent, so it can decide whether to approve it,
block it, or hold it for review.

Built as the project for [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## 📋 Rubric coverage

For peer review - where each grading criterion is evidenced in this repo:

| Criterion | Evidence |
|---|---|
| Problem description | [Problem description](#-problem-description) below. |
| EDA | `notebook.ipynb` sections 1-3: value ranges, missing values, duplicates, target distribution, `Amount`/`Time` analysis, per-class feature distributions, correlations, and feature importance (single-feature AUC, mutual information, correlation). Highlights in [EDA](#-eda-highlights). |
| Model training | `notebook.ipynb` sections 4-5: logistic regression, random forest and XGBoost, each tuned (grid search on validation, early stopping for XGBoost), then compared with 5-fold CV. See [Models](#-models-and-tuning). |
| Exporting notebook to script | `train.py` - trains the final model and saves it to `models/model.bin`. |
| Reproducibility | Dataset committed (`dataset/creditcard.csv.gz`), fixed seeds, `train.py` reproduces the notebook's numbers exactly. See [Data](#️-data) and [Setup](#️-setup). |
| Model deployment | `predict.py` - FastAPI web service. See [Run the service](#-run-the-service). |
| Dependency and environment management | `pyproject.toml` + `uv.lock`, virtual env in `.venv` - see [Setup](#️-setup). |
| Containerization | `Dockerfile` - build and run instructions in [Run with Docker](#-run-with-docker). |
| Cloud deployment | Google Cloud Run, live at <https://credit-card-fraud-gtnace3nia-ew.a.run.app/docs> - deployment code in `deploy/cloudrun.sh`, steps and test output in [Cloud deployment](#️-cloud-deployment-google-cloud-run). |

## 🎯 Problem description

Card fraud costs issuers, merchants and cardholders billions every year. Every
card payment needs an approve/decline decision within milliseconds, and the
fraudulent ones are a tiny fraction of all traffic. In this dataset, only
**0.17%** of transactions are fraud.

**Goal:** train a binary classifier that estimates, for a single transaction,
the probability that it is fraudulent.

**How it is used:** the model is deployed as a web service. The card
processing system calls it for every incoming transaction:

```
POST /predict  {"Time": 406, "V1": -2.31, ..., "V28": -0.14, "Amount": 0.0}
-> {"fraud_probability": 0.992, "is_fraud": true, "threshold": 0.703}
```

The caller then acts on the result. It can decline the payment, send it to
step-up authentication (e.g. 3-D Secure), or queue it for a fraud analyst.

**Why this is not a plain accuracy problem:**

- With 99.83% legitimate transactions, a model that never flags fraud is
  99.83% "accurate" and completely useless. Models are therefore compared
  with **PR-AUC (average precision)**, with ROC AUC reported alongside.
- The two errors have very different costs. A missed fraud (false negative)
  loses the full transaction amount. A false alarm (false positive) only
  causes friction for a real customer. The decision threshold behind
  `is_fraud` is therefore tuned (see [Models](#-models-and-tuning)), not left
  at 0.5.

## 🗄️ Data

**Source:** [OpenML dataset 1597 - "creditcard"](https://www.openml.org/d/1597),
originally released by the Machine Learning Group of Université Libre de
Bruxelles (ULB) in collaboration with Worldline:

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.

**Contents:** 284,807 transactions made by European cardholders over two days
in September 2013. Each row is one transaction:

| Column | Meaning |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset (0 - 172,792, i.e. ~48 h) |
| `V1` ... `V28` | Principal components (PCA) of the original features, anonymized for confidentiality |
| `Amount` | Transaction amount (EUR) |
| `Class` | **Target** - 1 = fraud, 0 = legitimate |

There are no missing values. There are 1,081 exact duplicate rows, which are
removed before splitting (283,726 rows remain, 473 of them fraud).

**Storage:**

- `dataset/creditcard.csv.gz` - the full dataset, gzip-compressed (~68 MB),
  **committed to this repo**. pandas reads it directly:
  `pd.read_csv("dataset/creditcard.csv.gz")`.
- `dataset/*.csv` is git-ignored, so an uncompressed copy (~144 MB, over
  GitHub's 100 MB file limit) never gets committed by accident.

**Re-downloading** (optional - only needed if the committed file is missing):

```bash
python scripts/download_data.py
```

The script uses only the Python standard library. It downloads the ARFF file
from OpenML, converts it to CSV (`Time, V1..V28, Amount, Class`), writes
`dataset/creditcard.csv.gz`, and checks the row and fraud counts
(284,807 / 492).

## 🔍 EDA highlights

Full analysis in `notebook.ipynb` (sections 1-3).

- **Imbalance:** 473 frauds in 283,726 transactions (0.167%) - hence PR-AUC.
- **Amount:** extremely right-skewed (median 22 EUR, max 25,691 EUR). Frauds
  are not generally larger: the median fraud is 9.82 EUR, and a quarter of
  frauds are ≤ 1 EUR (typical "card testing"). The model uses `log(1 + Amount)`.
- **Time:** raw `Time` is only meaningful within this dataset, so it is turned
  into **hour of day**. Fraud is far more common at night: ~1-1.5% of
  transactions at 2-4 am vs ~0.1% during the day.
- **Most informative features** (mutual information, single-feature AUC and
  correlation agree): `V14`, `V17`, `V12`, `V10`, `V11`, `V16`, `V3`, `V4`.
  `V13`, `V15`, `V22`, `V25` and `V26` carry almost no signal on their own.
  The PCA components are uncorrelated with each other.

## 🧠 Models and tuning

Data split: 60% train / 20% validation / 20% test, stratified (`common.py`).
Each model was tuned on the validation set, then the tuned models were compared
with **5-fold stratified CV on train+validation**. The test set was used only
once, at the very end.

| Model | Tuned | Best settings | Validation PR-AUC | 5-fold CV PR-AUC |
|---|---|---|---|---|
| Logistic regression | `C`, `class_weight` | `C=0.01`, no class weight | 0.799 | 0.759 ± 0.040 |
| Random forest | `max_depth`, `min_samples_leaf`, `class_weight`, `n_estimators` | depth 20, leaf 1, 200 trees | 0.877 | 0.850 ± 0.026 |
| **XGBoost** | `max_depth`, `learning_rate`, `min_child_weight`, `scale_pos_weight`, trees (early stopping) | depth 8, lr 0.05, `scale_pos_weight=10`, 165 trees | 0.893 | **0.852 ± 0.025** |

**Selected: XGBoost.** It is statistically tied with the random forest, but it
trains ~8x faster and its model file is ~6x smaller (0.7 MB vs 4.6 MB), which
suits a real-time service. Logistic regression has the *highest* ROC AUC
(0.984) yet the worst PR-AUC, which shows why ROC AUC misleads on imbalanced
data.

**Decision threshold:** 0.703, chosen to maximize F1 on out-of-fold CV
predictions.

**Final result on the held-out test set** (56,746 transactions, 95 frauds):

| PR-AUC | ROC AUC | Precision @ 0.703 | Recall @ 0.703 |
|---|---|---|---|
| 0.807 | 0.975 | 0.946 | 0.737 |

So 95% of the model's fraud alerts are real fraud, and it catches 74% of
frauds. A bank wanting to catch more fraud would lower the threshold and accept
more false alarms.

## ⚙️ Setup

Requires [uv](https://docs.astral.sh/uv/) (and Docker for the container).

```bash
uv sync                      # creates .venv and installs all dependencies from uv.lock
source .venv/bin/activate    # optional: activate the env (otherwise prefix commands with `uv run`)
```

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`:

- runtime (training + service): pandas, numpy, scikit-learn, xgboost, fastapi, uvicorn
- `dev` group (notebook, plots, test client, tests): jupyter, matplotlib,
  seaborn, requests, pytest, httpx - not installed in the Docker image

All commands below are run from the project root.

## 📓 Notebook

```bash
uv run jupyter notebook notebook.ipynb
```

The notebook is committed with its outputs. To re-execute it end to end
(~3 min on 16 cores):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

## 🏋️ Training

```bash
uv run python train.py
```

This trains the final XGBoost model, picks the threshold, prints the test
metrics and writes `models/model.bin` (~10 s). The trained model is committed,
so this step is optional before running the service.

## 🚀 Run the service

`predict.py` loads `models/model.bin` and serves it with FastAPI on port 9696:

| Endpoint | Description |
|---|---|
| `POST /predict` | One transaction (`Time`, `V1`..`V28`, `Amount`) -> `fraud_probability`, `is_fraud`, `threshold` |
| `GET /health` | Liveness check |
| `GET /docs` | Interactive API docs - try `/predict` in the browser (pre-filled with a real fraud example) |

### 🐳 Run with Docker

From the project root:

```bash
# 1. Build the image from the Dockerfile and name it "credit-card-fraud" (~1-2 min the first time)
docker build -t credit-card-fraud .

# 2. Run it: -p maps container port 9696 to localhost:9696, --rm removes the container on exit
docker run -it --rm -p 9696:9696 credit-card-fraud
```

Check it is up (in a second terminal) with `curl localhost:9696/health`, or
open <http://localhost:9696/docs>. Stop it with Ctrl+C. To run it in the
background instead, replace `-it` with `-d --name fraud` and stop it later
with `docker stop fraud`.

The image (~960 MB) contains only the runtime dependencies (installed from
`uv.lock`, so the same library versions the model was trained with),
`predict.py`, `common.py` and the model.

### 🐍 Run with Python

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

## 🧪 Using the API

With the service running (Docker or Python), in a second terminal:

```bash
uv run python predict_test.py
```

This sends 5 real fraud and 5 real legitimate transactions from the held-out
test set (never seen during training) and compares the predictions with the
truth. Output from the Docker container:

```
Service at http://localhost:9696: {'status': 'ok', 'model': 'xgboost'}

Amount      0.76 EUR | fraud_probability = 0.0006 | predicted: legit | actual: fraud | WRONG
Amount      1.18 EUR | fraud_probability = 0.9991 | predicted: FRAUD | actual: fraud | correct
Amount     65.00 EUR | fraud_probability = 0.9655 | predicted: FRAUD | actual: fraud | correct
Amount      0.00 EUR | fraud_probability = 0.0406 | predicted: legit | actual: fraud | WRONG
Amount     76.94 EUR | fraud_probability = 0.9835 | predicted: FRAUD | actual: fraud | correct
Amount     21.20 EUR | fraud_probability = 0.0001 | predicted: legit | actual: legit | correct
Amount      1.00 EUR | fraud_probability = 0.0003 | predicted: legit | actual: legit | correct
Amount     32.70 EUR | fraud_probability = 0.0000 | predicted: legit | actual: legit | correct
Amount    154.50 EUR | fraud_probability = 0.0000 | predicted: legit | actual: legit | correct
Amount     20.90 EUR | fraud_probability = 0.0001 | predicted: legit | actual: legit | correct

8/10 correct (threshold = 0.702853)
```

The two missed frauds are expected. At this threshold the model catches ~74%
of frauds (see [Models](#-models-and-tuning)). Use `--n 20` for a bigger
sample, or `--url` to point at another host.

Or call it directly with `curl`:

```bash
curl -s -X POST localhost:9696/predict -H 'Content-Type: application/json' -d '{
  "Time": 406, "V1": -2.3122, "V2": 1.952, "V3": -1.6099, "V4": 3.9979, "V5": -0.5222,
  "V6": -1.4265, "V7": -2.5374, "V8": 1.3917, "V9": -2.7701, "V10": -2.7723,
  "V11": 3.202, "V12": -2.8999, "V13": -0.5952, "V14": -4.2893, "V15": 0.3897,
  "V16": -1.1407, "V17": -2.8301, "V18": -0.0168, "V19": 0.417, "V20": 0.1269,
  "V21": 0.5172, "V22": -0.035, "V23": -0.4652, "V24": 0.3202, "V25": 0.0445,
  "V26": 0.1778, "V27": 0.2611, "V28": -0.1433, "Amount": 0
}'
```

Invalid input (a missing field, a negative amount, a non-number) is rejected
with HTTP 422 and a message saying which field is wrong.

### Tests

```bash
uv run pytest
```

API tests (`tests/test_predict.py`): health check, a known fraud is flagged,
an average transaction is not, and invalid input is rejected.

## ☁️ Cloud deployment (Google Cloud Run)

**Live service:** <https://credit-card-fraud-gtnace3nia-ew.a.run.app>

- Interactive API docs: <https://credit-card-fraud-gtnace3nia-ew.a.run.app/docs> - open `POST /predict`, click **Try it out**,
  then **Execute** (pre-filled with a real fraud transaction).
- Health check: <https://credit-card-fraud-gtnace3nia-ew.a.run.app/health>

> The service scales to zero when idle, so the first request after a quiet
> period can take a few seconds.

Test it with the same client as locally:

```bash
uv run python predict_test.py --url https://credit-card-fraud-gtnace3nia-ew.a.run.app
```

Output from the deployed service (identical to the local Docker run):

```
Service at https://credit-card-fraud-gtnace3nia-ew.a.run.app: {'status': 'ok', 'model': 'xgboost'}

Amount      0.76 EUR | fraud_probability = 0.0006 | predicted: legit | actual: fraud | WRONG
Amount      1.18 EUR | fraud_probability = 0.9991 | predicted: FRAUD | actual: fraud | correct
Amount     65.00 EUR | fraud_probability = 0.9655 | predicted: FRAUD | actual: fraud | correct
Amount      0.00 EUR | fraud_probability = 0.0406 | predicted: legit | actual: fraud | WRONG
Amount     76.94 EUR | fraud_probability = 0.9835 | predicted: FRAUD | actual: fraud | correct
Amount     21.20 EUR | fraud_probability = 0.0001 | predicted: legit | actual: legit | correct
Amount      1.00 EUR | fraud_probability = 0.0003 | predicted: legit | actual: legit | correct
Amount     32.70 EUR | fraud_probability = 0.0000 | predicted: legit | actual: legit | correct
Amount    154.50 EUR | fraud_probability = 0.0000 | predicted: legit | actual: legit | correct
Amount     20.90 EUR | fraud_probability = 0.0001 | predicted: legit | actual: legit | correct

8/10 correct (threshold = 0.702853)
```

Or with `curl` (the same request as in [Using the API](#-using-the-api), against
the cloud URL):

```bash
curl -s https://credit-card-fraud-gtnace3nia-ew.a.run.app/health
```

<!-- screenshot: images/cloudrun-predict-test.png -->

### How it is deployed

`deploy/cloudrun.sh` deploys with `gcloud run deploy --source .`: Cloud Build
builds the image from the `Dockerfile` in the cloud and stores it in Artifact
Registry, then Cloud Run serves it over HTTPS. No local Docker build or
manual image push is needed. `.gcloudignore` keeps the upload down to the
files the image needs.

**One-time setup:**

1. Create a Google Cloud account and a project with billing enabled (new
   accounts get free trial credit).
2. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and log in:
   ```bash
   gcloud init        # log in and pick the project
   ```

**Deploy, inspect, remove:**

```bash
./deploy/cloudrun.sh deploy    # enables the APIs, builds, deploys, prints the URL
./deploy/cloudrun.sh url       # prints the URL of the running service
./deploy/cloudrun.sh delete    # deletes the service and its images
```

Region and service name default to `europe-west1` / `credit-card-fraud`. You
can override them, e.g. `REGION=us-central1 ./deploy/cloudrun.sh deploy`.

### Cost and abuse protection

The endpoint is public so reviewers can test it without credentials. Cost is
capped in the deploy settings instead:

- `--max-instances 1` - at most one container, however many requests arrive.
- `--concurrency 20`, `--timeout 10` - bounded work per container.
- Scales to zero when idle (the default), so idle time costs nothing.
- 512 MiB memory - the container uses ~140 MiB.

With review-level traffic this stays within Cloud Run's free tier.

## 🗂️ Project layout

```
dataset/
  creditcard.csv.gz   # full dataset, gzip-compressed (committed)
scripts/
  download_data.py    # re-download from OpenML -> dataset/creditcard.csv.gz
models/
  model.bin           # trained pipeline + threshold (written by train.py, committed)
tests/
  test_predict.py     # API tests
common.py             # shared loading, cleaning, splitting, feature engineering
notebook.ipynb        # data prep, EDA, feature importance, model selection + tuning
train.py              # trains the final model -> models/model.bin
predict.py            # FastAPI web service
predict_test.py       # sends real test-set transactions to the service
deploy/
  cloudrun.sh         # deploy to / remove from Google Cloud Run
Dockerfile            # container for the web service
.dockerignore         # keeps the Docker build context small
.gcloudignore         # files not uploaded to Cloud Build
pyproject.toml        # dependencies (uv)
uv.lock               # pinned dependency versions
```

## ✅ Status

Done: problem definition, data preparation, EDA, model training and tuning,
training script, web service, Docker container, tests, and deployment to
Google Cloud Run (<https://credit-card-fraud-gtnace3nia-ew.a.run.app>).
