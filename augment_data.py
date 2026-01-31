import pandas as pd
import numpy as np
import random

# Load original dataset
# Trying both paths just in case, but assuming root execution
try:
    df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
except FileNotFoundError:
    df = pd.read_csv("app/data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(f"Original shape: {df.shape}")

# 1. Maternity/Paternity Leave (Boolean: 0 or 1)
# Logic: Higher probability if Female and Age between 25-40
def assign_mat_leave(row):
    prob = 0.02 # Base probability
    if row['Age'] >= 25 and row['Age'] <= 40:
        prob += 0.05
    if row['Gender'] == 'Female':
        prob += 0.05
    
    return 1 if random.random() < prob else 0

df['MaternityPaternityLeave'] = df.apply(assign_mat_leave, axis=1)

# 2. Workplace Harassment (Boolean: 0 or 1)
# Logic: Low probability (e.g. 5%), but strong correlation with Attrition
# We will "force" some correlation by checking Attrition status (for training data)
# But since this is synthesis for *new* features, we can just assign it and let the model learn the noise or 
# if we want to demonstrate it, we should make it correlated.
# If Harassment=1, increase change of Attrition='Yes'. 
# Since Attrition is already defined, we can't easily change it backwards without looking weird.
# Instead, let's just make Harassment more likely if Attrition is already Yes.
def assign_harassment(row):
    prob = 0.03 # Base chance
    if row['Attrition'] == 'Yes':
        prob = 0.30 # Much higher chance they left because of this
    return 1 if random.random() < prob else 0

df['WorkplaceHarassment'] = df.apply(assign_harassment, axis=1)

# 3. Remote Work Frequency (0 to 5 days a week)
# Logic: Random, maybe slightly correlated with JobRole or Department but random is fine.
# If RemoteWork is high, maybe Attrition is lower?
def assign_remote_work(row):
    # If Attrition is No, maybe likely to have more remote work (happier)?
    # Let's just make it random distribution
    weights = [0.4, 0.2, 0.15, 0.1, 0.1, 0.05] # weighted towards 0, 1, 2 days
    return np.random.choice([0, 1, 2, 3, 4, 5], p=weights)

df['RemoteWorkFrequency'] = df.apply(assign_remote_work, axis=1)

# 4. Mental Health Resources (Boolean: 0=No, 1=Yes)
# Logic: If No, slightly higher chance of Attrition
df['MentalHealthResources'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])

# 5. Project Deadline Pressure (1=Low, 2=Medium, 3=High, 4=Extreme)
# Logic: Random distribution
df['ProjectDeadlinePressure'] = np.random.choice([1, 2, 3, 4], size=len(df), p=[0.2, 0.4, 0.3, 0.1])

# 6. Skill Development Hours Per Month (0 to 20)
# Logic: Normal distribution around 5 hours
df['SkillDevelopmentHours'] = np.random.normal(5, 3, size=len(df)).astype(int)
df['SkillDevelopmentHours'] = df['SkillDevelopmentHours'].clip(0, 20)

# Drop redundant columns that hinder the model or are useless
redundant_cols = ['Over18', 'StandardHours', 'EmployeeCount']
df.drop(columns=[c for c in redundant_cols if c in df.columns], inplace=True)
# Note: Keeping EmployeeNumber as ID might be useful for tracking but not for training. Model script usually drops it or we should.

print("New columns added: MentalHealthResources, ProjectDeadlinePressure, SkillDevelopmentHours")
print(f"Removed redundant columns: {redundant_cols}")

print("New columns added: MaternityPaternityLeave, WorkplaceHarassment, RemoteWorkFrequency")
print(f"New shape: {df.shape}")

# Save augmented data
# We save to both locations to be safe/consistent
output_path_1 = "data/HR_Employee_Attrition_Augmented.csv"
output_path_2 = "app/data/HR_Employee_Attrition_Augmented.csv"

df.to_csv(output_path_1, index=False)
try:
    df.to_csv(output_path_2, index=False)
except FileNotFoundError:
    pass

print(f"Saved augmented dataset to {output_path_1}")
