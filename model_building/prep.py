import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/tourism.csv")

# Drop the customer Id 
df.drop(columns=["CustomerID"], inplace=True) 

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("data/Xtrain.csv", index=False)
Xtest.to_csv("data/Xtest.csv", index=False)
ytrain.to_csv("data/ytrain.csv", index=False)
ytest.to_csv("data/ytest.csv", index=False)

print("Data prepared: train/test splits written.")
