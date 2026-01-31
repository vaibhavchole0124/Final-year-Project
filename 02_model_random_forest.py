import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import joblib

print("▶ Starting Random Forest training script...")

# -----------------------------
# 1. Load Dataset
# -----------------------------
pd.set_option('display.max_columns', None)

data_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(data_path)

print("Shape of dataset:", df.shape)

# -----------------------------
# 2. Prepare Features & Target
# -----------------------------
data = df.copy()
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

X = data.drop(columns=['Attrition'])
y = data['Attrition']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# -----------------------------
# 3. Train–Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# -----------------------------
# 4. Identify Numeric & Categorical Columns
# -----------------------------
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("\nCategorical columns:", categorical_cols)

# -----------------------------
# 5. Preprocessing (Scaling + One-Hot Encoding)
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print("\nPreprocessed train shape:", X_train_prep.shape)
print("Preprocessed test shape:", X_test_prep.shape)

# -----------------------------
# 6. Handle Imbalance with SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
print("\nBefore SMOTE class counts:", np.bincount(y_train))

X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)

print("After SMOTE class counts:", np.bincount(y_train_res))

# -----------------------------
# 7. Train Random Forest Model
# -----------------------------
rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=None
)

rf_clf.fit(X_train_res, y_train_res)

y_pred = rf_clf.predict(X_test_prep)

acc = accuracy_score(y_test, y_pred)
print("\nRandom Forest Accuracy:", round(acc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 8. Save Preprocessor & Model
# -----------------------------
os.makedirs("models", exist_ok=True)

preprocessor_path = os.path.join("models", "preprocessor_rf.pkl")
model_path = os.path.join("models", "random_forest_model.pkl")

joblib.dump(preprocessor, preprocessor_path)
joblib.dump(rf_clf, model_path)

print(f"\nSaved RF preprocessor to: {preprocessor_path}")
print(f"Saved RF model to: {model_path}")

print("\n✅ Random Forest model training complete.")
