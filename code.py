# Customer Churn Prediction
# Machine Learning Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load Dataset

df = pd.read_csv("customerchurn.csv", sep="\t")
print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 2. Data Cleaning
# Convert TotalCharges to numeric
# Some datasets contain blank values in this column

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Remove rows containing missing values
df = df.dropna()

# Remove customer ID because it does not help prediction
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# 3. Convert Target Column

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# 4. Exploratory Data Analysis

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

sns.countplot(x="Churn", data=df)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()

# 5. Separate Features and Target

X = df.drop("Churn", axis=1)
y = df["Churn"]

# 6. Convert Categorical Features

categorical_columns = X.select_dtypes(
    include=["str"]
).columns

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

# 7. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# 8. Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train Machine Learning Model

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# 10. Make Predictions

y_pred = model.predict(X_test)

# 11. Model Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 12. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 13. Predict a New Customer

# Example:
# 0 = customer will NOT churn
# 1 = customer WILL churn

sample_customer = X.iloc[[0]]

prediction = model.predict(
    scaler.transform(sample_customer)
)

if prediction[0] == 1:
    print("\nPrediction: Customer is likely to CHURN.")
else:
    print("\nPrediction: Customer is likely to STAY.")
