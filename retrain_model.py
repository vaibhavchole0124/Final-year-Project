import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Prepare target
y = (df["Attrition"] == "Yes").astype(int)
X = df.drop(columns=["Attrition"])

# Split numeric & categorical
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Build preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Build model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf.fit(X_train, y_train)

# Save updated model
joblib.dump(clf["model"], "models/random_forest_model.pkl")
joblib.dump(clf["preprocessor"], "models/preprocessor_rf.pkl")

print("🎉 MODEL RETRAINED SUCCESSFULLY!")
