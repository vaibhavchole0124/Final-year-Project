"""
03_model_comparison.py

Compare multiple ML models for Employee Attrition Prediction:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Outputs:
- Metrics printed in console
- CSV file: models/model_comparison_results.csv
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

print("📥 Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# Target variable: Attrition (Yes/No)
y = df["Attrition"].map({"Yes": 1, "No": 0})  # convert to 1/0
X = df.drop("Attrition", axis=1)

# ---------------------------------------------------
# 2. Identify numeric & categorical columns
# ---------------------------------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ---------------------------------------------------
# 3. Preprocessing Pipeline
# ---------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ---------------------------------------------------
# 4. Define Models to Compare
# ---------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "SVM (RBF Kernel)": SVC(
        probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42
    ),
}

# ---------------------------------------------------
# 5. Train/Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# ---------------------------------------------------
# 6. Train & Evaluate Each Model
# ---------------------------------------------------
results = []

for name, clf in models.items():
    print("\n" + "=" * 60)
    print(f"🚀 Training model: {name}")
    print("=" * 60)

    # Create full pipeline: preprocessor + classifier
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    # Train
    model_pipeline.fit(X_train, y_train)

    # Predict
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro)   : {rec:.4f}")
    print(f"F1-score (macro) : {f1:.4f}")
    print(f"ROC-AUC          : {roc:.4f}")

    results.append(
        {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision_macro": round(prec, 4),
            "Recall_macro": round(rec, 4),
            "F1_macro": round(f1, 4),
            "ROC_AUC": round(roc, 4),
        }
    )

# ---------------------------------------------------
# 7. Save Results to CSV
# ---------------------------------------------------
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison Results:")
print(results_df)

# Ensure models folder exists
import os

os.makedirs("models", exist_ok=True)
output_path = "models/model_comparison_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\n✅ Saved comparison results to: {output_path}")
