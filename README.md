# MLOps Breast Cancer Classifier

A complete beginner-friendly **MLOps project** you can push to GitHub.

It includes:
- End-to-end ML pipeline
- Data ingestion and train/test split
- Model training with `RandomForestClassifier`
- Evaluation metrics logging with **MLflow**
- Model serialization with `joblib`
- **FastAPI** inference API
- **Pytest** unit tests
- **GitHub Actions** CI pipeline
- **Docker** support

## Project Structure

```bash
mlops-breast-cancer-project/
├── app/
│   └── main.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_ingestion.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
├── models/
│   └── model.joblib
├── tests/
│   ├── test_api.py
│   └── test_train.py
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── requirements.txt
├── .gitignore
└── run_pipeline.py
```

## 1. Create virtual environment

```bash
python -m venv .venv
```

### Windows
```bash
.venv\Scripts\activate
```

### Linux / macOS
```bash
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run training pipeline

```bash
python run_pipeline.py
```

This will:
- create dataset files in `data/`
- train the model
- evaluate it
- save the model in `models/model.joblib`
- log experiment info in MLflow

## 4. Run MLflow UI

```bash
mlflow ui
```

Then open the local MLflow UI in your browser.

## 5. Start API server

```bash
uvicorn app.main:app --reload
```

## 6. Test API

Open:
- `http://127.0.0.1:8000/docs`

## Example Prediction Request

```json
{
  "features": [
    14.0, 20.0, 90.0, 600.0, 0.1,
    0.15, 0.2, 0.1, 0.2, 0.06,
    0.5, 1.2, 3.5, 40.0, 0.005,
    0.02, 0.03, 0.01, 0.02, 0.003,
    15.0, 25.0, 100.0, 700.0, 0.14,
    0.3, 0.4, 0.15, 0.3, 0.08
  ]
}
```

## Run Tests

```bash
pytest -v
```

## Run with Docker

```bash
docker build -t mlops-breast-cancer .
docker run -p 8000:8000 mlops-breast-cancer
```

## GitHub Push Commands

```bash
git init
git add .
git commit -m "Initial commit - MLOps project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Tech Stack
- Python
- scikit-learn
- pandas
- FastAPI
- MLflow
- pytest
- Docker
- GitHub Actions
# safety

