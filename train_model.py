
# ============================================================
# train_model.py
# AI-ML Assignment 10 : Heart Disease Prediction - Model Training
# Dataset: Kaggle - johnsmith88/heart-disease-dataset
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


DATA_PATH = "heart.csv"

# ============================================================
# TASK 1: DATA UNDERSTANDING AND PREPROCESSING
# ============================================================
print("=" * 60)
print("TASK 1: DATA UNDERSTANDING AND PREPROCESSING")
print("=" * 60)

# 1. Load the dataset
df = pd.read_csv(DATA_PATH)

# 2. Display first five records
print("\nFirst five records:")
print(df.head())

# 3. Identify numerical features and target variable
target_col = "target"  # this dataset's label column
numerical_features = [c for c in df.columns if c != target_col]

print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
print(f"Target variable: '{target_col}'")
print(f"\nTarget value counts:\n{df[target_col].value_counts()}")

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print("\nDropped rows with missing values. New shape:", df.shape)
else:
    print("\nNo missing values found.")

# 5. Split into 80% train / 20% test
X = df[numerical_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Feature scaling (helps Logistic Regression converge well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# TASK 2: MODEL DEVELOPMENT
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: MODEL DEVELOPMENT")
print("=" * 60)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel: Logistic Regression")
print(f"Test Accuracy: {accuracy:.4f}")

# Save the trained model AND the scaler AND the feature order
# (all three are needed at inference time in the Flask API)
joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "feature_names": numerical_features,
    },
    "model.pkl",
)

print("\nSaved trained model, scaler, and feature list to 'model.pkl'")
print("\nDone. You can now run app.py to serve predictions.")
