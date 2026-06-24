# Customer Churn Prediction MLOps Pipeline

## Overview

An end-to-end Machine Learning pipeline for predicting customer churn using Random Forest, integrated with DVC for data and pipeline versioning and MLflow for experiment tracking.

This project demonstrates reproducible machine learning workflows, experiment management, model versioning, and MLOps best practices.

---

## Features

- End-to-End ML Pipeline
- Data Versioning with DVC
- Experiment Tracking with MLflow
- Automated Pipeline Reproduction
- Hyperparameter Management using YAML
- Model Evaluation and Metrics Logging
- Reproducible Training Workflow
- Git-based Version Control

---

## Project Architecture

```text
Raw Dataset
     │
     ▼
Data Ingestion
     │
     ▼
Preprocessing
     │
     ├── train.csv
     └── test.csv
     │
     ▼
Model Training
(Random Forest)
     │
     ├── rf_model.pkl
     └── MLflow Tracking
     │
     ▼
Evaluation
     │
     ├── metrics.json
     └── MLflow Metrics
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- DVC
- MLflow
- Git

---

## Pipeline Stages

### Data Preprocessing

- Data cleaning
- Encoding categorical features
- Train-test split
- Export processed datasets

### Model Training

- Random Forest Classifier
- Hyperparameters loaded from `params.yaml`
- Model saved as `rf_model.pkl`
- Parameters logged to MLflow

### Evaluation

- Accuracy
- Precision
- Recall
- F1 Score

Metrics are stored in:

```text
reports/metrics.json
```

and tracked using MLflow.

---

## Running the Project

### Run Pipeline

```bash
dvc repro
```

---

## MLflow Experiment Tracking

Launch MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Track:

- Parameters
- Metrics
- Model Artifacts
- Experiment Runs

---

## DVC Commands

### Reproduce Pipeline

```bash
dvc repro
```

### Run Experiments

```bash
dvc exp run
```

### Compare Experiments

```bash
dvc exp show
```

### Push Data to Remote Storage

```bash
dvc push
```

### Pull Data from Remote Storage

```bash
dvc pull
```

---

## Results

The model performance metrics are automatically generated after evaluation and stored in:

```text
reports/metrics.json
```

Example:

```json
{
    "accuracy": 0.81,
    "precision": 0.74,
    "recall": 0.67,
    "f1_score": 0.70
}
```

---

## Future Improvements

- FastAPI Deployment
- Docker Containerization
- GitHub Actions CI/CD
- Model Registry
- Streamlit Dashboard
- Cloud Deployment (AWS/Azure/GCP)
