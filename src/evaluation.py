import json
import joblib
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
test = pd.read_csv("data/processed/test.csv")
X_test = test.drop("Churn",axis=1)
y_test = test["Churn"]
model = joblib.load("models/rf_model.pkl")
pred = model.predict(X_test)
acc = accuracy_score(y_test,pred)
precision = precision_score(y_test,pred)
recall = recall_score(y_test,pred)
f1 = f1_score(y_test,pred)
metrics = {
    "accuracy":float(acc),
    "precision":float(precision),
    "recall":float(recall),
    "f1_score":float(f1)
}
with open(
    "reports/metrics.json",
    "w"
) as f:
    json.dump(metrics,f,indent=4)
mlflow.set_experiment("Customer Churn Prediction")
with mlflow.start_run():
    mlflow.log_metric("accuracy",acc)
    mlflow.log_metric('precision',precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1)
print(metrics)
print("Evaluation donee!!")