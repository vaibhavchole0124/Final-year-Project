import pandas as pd
import numpy as np
import random

def generate_bulk_data(num_samples=50):
    # Seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    data = {
        "EmployeeNumber": np.arange(1001, 1001 + num_samples),
        "Age": np.random.randint(18, 60, num_samples),
        "BusinessTravel": np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], num_samples),
        "DailyRate": np.random.randint(100, 1500, num_samples),
        "Department": np.random.choice(['Sales', 'Research & Development', 'Human Resources'], num_samples),
        "DistanceFromHome": np.random.randint(1, 30, num_samples),
        "Education": np.random.randint(1, 6, num_samples),
        "EducationField": np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'], num_samples),
        "EmployeeCount": [1] * num_samples,
        "EmployeeNumber": np.arange(1000, 1000 + num_samples),
        "EnvironmentSatisfaction": np.random.randint(1, 5, num_samples),
        "Gender": np.random.choice(['Male', 'Female'], num_samples),
        "HourlyRate": np.random.randint(30, 100, num_samples),
        "JobInvolvement": np.random.randint(1, 5, num_samples),
        "JobLevel": np.random.randint(1, 6, num_samples),
        "JobRole": np.random.choice([
            'Sales Executive', 'Research Scientist', 'Laboratory Technician', 
            'Manufacturing Director', 'Healthcare Representative', 'Manager', 
            'Sales Representative', 'Research Director', 'Human Resources'
        ], num_samples),
        "JobSatisfaction": np.random.randint(1, 5, num_samples),
        "MaritalStatus": np.random.choice(['Single', 'Married', 'Divorced'], num_samples),
        "MonthlyIncome": np.random.randint(2000, 20000, num_samples),
        "MonthlyRate": np.random.randint(2000, 27000, num_samples),
        "NumCompaniesWorked": np.random.randint(0, 10, num_samples),
        "Over18": ['Y'] * num_samples,
        "OverTime": np.random.choice(['Yes', 'No'], num_samples),
        "PercentSalaryHike": np.random.randint(11, 26, num_samples),
        "PerformanceRating": np.random.choice([3, 4], num_samples), # Usually 3 or 4 in this dataset
        "RelationshipSatisfaction": np.random.randint(1, 5, num_samples),
        "StandardHours": [80] * num_samples,
        "StockOptionLevel": np.random.randint(0, 4, num_samples),
        "TotalWorkingYears": np.random.randint(0, 40, num_samples),
        "TrainingTimesLastYear": np.random.randint(0, 7, num_samples),
        "WorkLifeBalance": np.random.randint(1, 5, num_samples),
        "YearsAtCompany": np.random.randint(0, 40, num_samples), # Clipped later
        "YearsInCurrentRole": np.zeros(num_samples), # Calc later
        "YearsSinceLastPromotion": np.zeros(num_samples), # Calc later
        "YearsWithCurrManager": np.zeros(num_samples), # Calc later
        
        # New Features
        "MaternityPaternityLeave": np.random.choice([0, 1], num_samples, p=[0.95, 0.05]),
        "WorkplaceHarassment": np.random.choice([0, 1], num_samples, p=[0.95, 0.05]),
        "RemoteWorkFrequency": np.random.randint(0, 6, num_samples),
        
        # Round 2 Features
        "MentalHealthResources": np.random.choice([0, 1], num_samples, p=[0.4, 0.6]),
        "ProjectDeadlinePressure": np.random.choice([1, 2, 3, 4], num_samples, p=[0.2, 0.4, 0.3, 0.1]),
        "SkillDevelopmentHours": np.random.randint(0, 20, num_samples)
    }

    df = pd.DataFrame(data)

    # logical consistency fixes
    df['YearsAtCompany'] = df.apply(lambda x: min(x['YearsAtCompany'], x['TotalWorkingYears']), axis=1)
    df['YearsInCurrentRole'] = df.apply(lambda x: min(np.random.randint(0, x['YearsAtCompany'] + 1), x['YearsAtCompany']), axis=1)
    df['YearsSinceLastPromotion'] = df.apply(lambda x: min(np.random.randint(0, x['YearsAtCompany'] + 1), x['YearsAtCompany']), axis=1)
    df['YearsWithCurrManager'] = df.apply(lambda x: min(np.random.randint(0, x['YearsAtCompany'] + 1), x['YearsAtCompany']), axis=1)

    # Force some edge cases for diversity test
    
    # Case 1: High Risk (Low Satisfaction, Overtime, Low Pay)
    df.loc[0, 'JobSatisfaction'] = 1
    df.loc[0, 'EnvironmentSatisfaction'] = 1
    df.loc[0, 'OverTime'] = 'Yes'
    df.loc[0, 'WorkplaceHarassment'] = 1
    df.loc[0, 'ProjectDeadlinePressure'] = 4
    df.loc[0, 'MonthlyIncome'] = 2500
    
    # Case 2: Low Risk (High Satisfaction, Good Pay)
    df.loc[1, 'JobSatisfaction'] = 4
    df.loc[1, 'EnvironmentSatisfaction'] = 4
    df.loc[1, 'OverTime'] = 'No'
    df.loc[1, 'WorkplaceHarassment'] = 0
    df.loc[1, 'ProjectDeadlinePressure'] = 1
    df.loc[1, 'MonthlyIncome'] = 15000
    df.loc[1, 'RemoteWorkFrequency'] = 5

    # Case 3: Maternity Leave Returnee
    df.loc[2, 'Gender'] = 'Female'
    df.loc[2, 'Age'] = 32
    df.loc[2, 'MaternityPaternityLeave'] = 1
    
    return df

if __name__ == "__main__":
    df = generate_bulk_data(50)
    output_path = "data/bulk_test_sample.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} diverse samples in {output_path}")
