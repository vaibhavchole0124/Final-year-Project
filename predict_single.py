import pandas as pd
import joblib

# 1. Load saved preprocessor and model (RANDOM FOREST)
preprocessor_path = "models/preprocessor_rf.pkl"
model_path = "models/random_forest_model.pkl"

preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)

print("✅ Loaded preprocessor and Random Forest model.")

# 2. Define ONE employee's data (example)
#    👉 Later this will come from frontend form / Excel
employee_data = {
    "Age": 29,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 5,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EmployeeCount": 1,
    "EmployeeNumber": 9999,  # dummy
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 70,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 3,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5000,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "Over18": "Y",
    "OverTime": "Yes",
    "PercentSalaryHike": 13,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StandardHours": 80,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 6,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 2,
    "YearsAtCompany": 3,
    "YearsInCurrentRole": 2,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2,
}

# 3. Convert to DataFrame (single row)
employee_df = pd.DataFrame([employee_data])

print("\nEmployee data:")
print(employee_df)

# 4. Preprocess using the same preprocessor as training
X_prep = preprocessor.transform(employee_df)

# 5. Predict
pred_class = model.predict(X_prep)[0]             # 0 or 1
pred_proba = model.predict_proba(X_prep)[0, 1]    # probability of attrition (class 1)

# 6. Interpret result
label = "Likely to LEAVE" if pred_class == 1 else "Likely to STAY"
risk_score = round(pred_proba * 100, 2)

print("\n🔍 Prediction Result:")
print(f"Prediction: {label}")
print(f"Attrition Probability: {pred_proba:.4f}")
print(f"Risk Score (0–100): {risk_score}")
