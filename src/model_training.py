import pandas as pd
import yaml
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
os.makedirs("models", exist_ok=True)
train = pd.read_csv("data/processed/train.csv")
X=train.drop("Churn",axis=1)
y=train["Churn"]

with open("params.yaml","r") as f:
    params = yaml.safe_load(f)
n_estimators = params["n_estimators"]
max_depth = params["max_depth"]
mlflow.set_experiment("Customer Churn Prediction")
with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X,y)
    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_param("max_depth",max_depth)
    model_path = "models/rf_model.pkl"
    joblib.dump(model,model_path)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )
    print(f"Model save {model_path}")
print("Trained!")