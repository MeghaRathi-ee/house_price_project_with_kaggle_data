# House Price Prediction — End-to-End MLOps Project

A production-grade MLOps pipeline for house price prediction, built from scratch using industry-standard tools.

---

## Project Overview

This project demonstrates a complete MLOps lifecycle — from raw data ingestion to automated retraining — using a synthetic house price dataset. The focus is on the **infrastructure and pipeline architecture**, not model performance (the dataset is synthetic with no real signal).

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data versioning | DVC + Azure Blob Storage |
| Experiment tracking | MLFlow |
| Model serving | FastAPI + Uvicorn |
| Containerization | Docker |
| Drift monitoring | Evidently AI |
| Pipeline orchestration | Apache Airflow |
| CI/CD | GitHub Actions |
| Cloud storage | Azure Blob Storage |

---

## Project Structure

```
house_price_mlops/
├── src/
│   ├── ingest.py          # Load raw CSV, train/test split
│   ├── preprocess.py      # Feature engineering, scaling, encoding
│   ├── train.py           # Train 8 models, MLFlow tracking
│   ├── evaluate.py        # Metrics computation, reverse log transform
│   └── monitor.py         # Evidently drift detection + retrain trigger
├── app/
│   └── main.py            # FastAPI prediction endpoint
├── dags/
│   └── pipeline.py        # Airflow DAG for weekly automation
├── tests/
│   └── test_pipeline.py   # 8 pytest tests
├── .github/workflows/
│   └── ci.yaml            # GitHub Actions CI/CD
├── dvc.yaml               # DVC pipeline stages
├── params.yaml            # All hyperparameters and config
├── Dockerfile             # Container definition
└── docker-compose.yml     # Local deployment
```

---

## ML Pipeline

```
raw CSV → ingest → preprocess → train → evaluate → monitor → retrain trigger
```

### Stage details

**ingest.py** — Loads raw CSV, drops `Id` column, splits into 80/20 train/test, saves to `data/processed/`.

**preprocess.py** — Six preprocessing steps:
1. Drop useless columns (Id)
2. Outlier removal via IQR method
3. Feature engineering — `HouseAge`, `TotalRooms`, `AreaPerRoom`, `IsNew`
4. Log transform on target (Price)
5. Correlation-based feature selection
6. RobustScaler + OneHotEncoder via sklearn Pipeline

**train.py** — Trains 8 models and logs each to MLFlow:
- LinearRegression, Ridge, Lasso, ElasticNet (linear baselines)
- DecisionTree, RandomForest (tree models)
- XGBoost, LightGBM (gradient boosting)

Best model auto-selected by R2 and saved to `model.pkl`. Registered in MLFlow Model Registry → Staging.

**evaluate.py** — Loads best model, reverses log transform (`np.expm1`), computes RMSE, MAE, R2, MAPE on actual price scale.

**monitor.py** — Runs Evidently AI drift report comparing training data vs new incoming data. Fires retraining trigger if drift share > 5% or critical features drifted.

---

## Model Comparison (latest run)

| Model | R2 | RMSE | MAPE |
|-------|----|------|------|
| Lasso | -0.134 | 297,023 | 88.5% ← BEST |
| ElasticNet | -0.134 | 297,023 | 88.5% |
| Ridge | -0.137 | 297,417 | 88.8% |
| LinearRegression | -0.137 | 297,425 | 88.8% |
| DecisionTree | -0.162 | 300,639 | 88.7% |
| RandomForest | -0.201 | 305,634 | 89.7% |

> **Note:** All R2 values are negative because the dataset is synthetically generated — Price has near-zero correlation with all features (confirmed via EDA). Linear models outperform tree models because they don't overfit to noise. The pipeline architecture is production-grade regardless of model performance.

---

## Running the Pipeline

```bash
# Setup
conda create -n mlops_env python=3.10
conda activate mlops_env
pip install -r requirements.txt

# Initialize DVC
dvc init
dvc remote add -d azure azure://dvcstore/hpp

# Run full pipeline
dvc repro

# Check metrics
dvc metrics show
dvc metrics diff   # compare with previous run
```

---

## FastAPI Serving

```bash
uvicorn app.main:app --reload --port 8000
```

**Predict endpoint:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Area": 2500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Floors": 2,
    "YearBuilt": 1990,
    "Location": "Downtown",
    "Condition": "Good",
    "Garage": "Yes"
  }'
```

**Response:**
```json
{
  "predicted_price": 441553.1,
  "predicted_price_fmt": "$441,553",
  "model_used": "Lasso",
  "features_used": {
    "HouseAge": 36,
    "TotalRooms": 5,
    "AreaPerRoom": 500.0,
    "IsNew": 0,
    "warnings": []
  }
}
```

---

## Docker

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

---

## MLFlow Tracking

```bash
mlflow ui
# Open http://localhost:5000
```

View all 8 model runs, compare metrics, inspect model artifacts, and promote best model to Production via the Model Registry.

---

## Drift Monitoring

Evidently AI compares training data distribution vs new incoming data. Generates an HTML report at `data/reports/drift_report.html`.

Retrain trigger fires when:
- Share of drifted columns > 5% threshold
- OR critical features (Area, YearBuilt, Price) drift detected

---

## Airflow Automation

Weekly pipeline runs automatically via Airflow DAG:

```
ingest → preprocess → train → evaluate → monitor → check_trigger
                                                         ↓
                                          retrain_notify OR no_retrain_notify
                                                         ↓
                                                     dvc_push
```

```bash
export AIRFLOW_HOME=~/airflow
airflow standalone
# Open http://localhost:8080
```

---

## CI/CD

GitHub Actions workflow runs on every push to `main`:
1. **Test job** — runs 8 pytest tests
2. **Pipeline job** — pulls data from Azure, runs `dvc repro`, pushes artifacts
3. **Docker job** — builds Docker image, runs health check

---

## Tests

```bash
pytest tests/ -v
```

8 tests covering: params validation, feature engineering correctness, model/preprocessor existence, metrics format, FastAPI schema validation, log transform reversibility.

---

## Dataset

**Source:** Synthetic house price dataset from Kaggle  
**Columns:** Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage → Price  
**Size:** 2000 rows, 10 columns, 0 missing values

---

## Key MLOps Concepts Demonstrated

- **Reproducibility** — DVC tracks exact data and model versions. `dvc repro` always produces identical results.
- **Experiment tracking** — Every model run logged with params, metrics, and artifacts in MLFlow.
- **Model registry** — Best model versioned and promoted through Staging → Production workflow.
- **Data versioning** — Dataset stored in Azure Blob Storage, referenced by hash in `dvc.lock`.
- **Drift detection** — Evidently AI detects when incoming data shifts from training distribution.
- **Automated retraining** — Airflow triggers weekly pipeline with branch logic for retrain vs skip.
- **CI/CD** — GitHub Actions validates every commit with tests and full pipeline run.