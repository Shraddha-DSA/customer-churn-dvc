import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("data/processed/train.csv")
X=train.drop("Churn",axis=1)
y=train["Churn"]
params=yaml.safe_load(
    open("params.yaml")
)
model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=42
)
model.fit(X,y)
joblib.dump(model,"models/rf_model.pkl")