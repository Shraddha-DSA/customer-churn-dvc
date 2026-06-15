import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("data/raw/churn.csv")
df.drop("CustomerID",axis=1,inplace=True)
for col in df.select_dtypes("object"):
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
train,test = train_test_split(
    df,test_size=0.2,random_state=42
)
train.to_csv("data/processed/train.csv",index=False)
test.to_csv("data/processed/test.csv",index=False)