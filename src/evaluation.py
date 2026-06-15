import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
test = pd.read_csv("data/processed/test.csv")
X_test = test.drop("Churn",axis=1)
y_test = test["Churn"]
model = joblib.load("models/rf_model.pkl")
pred = model.predict(X_test)
acc = accuracy_score(y_test,pred)
metrics = {
    "accuracy":float(acc)
}
with open(
    "reports/metrics.json",
    "w"
) as f:
    json.dump(metrics,f)