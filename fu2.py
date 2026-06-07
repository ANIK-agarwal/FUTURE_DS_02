# Customer Retention & Churn Analysis
# Author: Anika (2026 Internship Task 2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load & Inspect Data
# -----------------------------
# Replace with your dataset path
df = pd.read_csv("ar2.csv")

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# -----------------------------
# 2. Data Cleaning
# -----------------------------
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# Convert categorical columns to category type
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category")

# -----------------------------
# 3. Churn Rate Analysis
# -----------------------------
churn_rate = df['Churn'].value_counts(normalize=True) * 100
print("\nChurn Rate (%):\n", churn_rate)

sns.countplot(x='Churn', data=df, palette="Set2")
plt.title("Customer Churn Distribution")
plt.show()

# -----------------------------
# 4. Cohort Analysis (Signup Month)
# -----------------------------
# Example: If dataset has 'tenure' (months active)
df['Cohort'] = pd.cut(df['tenure'], bins=[0,12,24,36,48,60,72], 
                      labels=['0-12m','12-24m','24-36m','36-48m','48-60m','60-72m'])

cohort_churn = df.groupby('Cohort')['Churn'].value_counts(normalize=True).unstack()
print("\nCohort Churn Rates:\n", cohort_churn)

cohort_churn.plot(kind='bar', stacked=True, colormap="coolwarm")
plt.title("Churn by Customer Cohort")
plt.ylabel("Proportion")
plt.show()

# -----------------------------
# 5. Retention Drivers
# -----------------------------
# Example: Churn by Contract Type
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
print("\nChurn by Contract Type:\n", contract_churn)

contract_churn.plot(kind='bar', stacked=True, colormap="viridis")
plt.title("Churn by Contract Type")
plt.show()

# -----------------------------
# 6. Customer Lifetime Metrics
# -----------------------------
avg_tenure = df[df['Churn']=="No"]['tenure'].mean()
print("\nAverage Customer Lifetime (months):", round(avg_tenure,2))

# -----------------------------
# 7. Insights & Recommendations
# -----------------------------
print("\n📊 Insights:")
print("- Overall churn rate is {:.2f}%".format(churn_rate['Yes']))
print("- Customers with month-to-month contracts churn more often.")
print("- Longer-tenure cohorts show higher retention.")
print("- Average customer lifetime is ~{} months.".format(round(avg_tenure,2)))

print("\n💡 Recommendations:")
print("- Encourage customers to switch from month-to-month to annual contracts.")
print("- Offer loyalty rewards for customers crossing 12 months tenure.")
print("- Focus retention campaigns on early-tenure cohorts (0-12 months).")
