# Loan-Amount-Prediction-Portfolio-Analysis
This project analyzes a bank’s loan portfolio to identify the key factors that influence how much customers request in loans. Using 931 customer records, we apply statistical analysis and predictive modeling techniques to determine which variables meaningfully explain loan demand and how these insights can support better lending decisions.

---

## Business Question
**What factors most strongly drive the loan amounts requested by customers?**

---

## Dataset
- **Observations:** 931 customers  
- **Target Variable:** Loan Amount  
- **Candidate Predictors:**
  - Property value
  - Debt-to-income ratio
  - Existing mortgage balance
  - Years on job
  - Occupation category
  - Credit inquiries and credit history metrics

---

## Methodology
The analysis applies the following techniques:
- Correlation analysis
- Simple linear regression
- Multiple linear regression
- Logistic regression (high-value loans > $20,000)
- Stepwise variable selection
- Multicollinearity diagnostics (VIF)

---

## Key Findings

### 1. Property Value Is the Primary Driver
- Property value shows the strongest correlation with loan amount (**r = 0.46**)
- Existing mortgage follows at **r = 0.36**
- Credit inquiries, credit age, and job tenure show weak relationships

**Interpretation:**  
For every $10,000 increase in property value, customers request approximately $846 more in loan amount.

---

### 2. Self-Employed Customers Request Larger Loans
A logistic regression model was used to identify customers requesting high-value loans (>$20,000), which represent 37% of the portfolio.

- Self-employed customers are **2.4× more likely** to request high-value loans than managers
- Sales and “other” occupations are significantly less likely
- Debt-to-income ratio increases high-loan odds by ~2% per unit

**Model Performance:**
- Accuracy: 73%
- Precision: 74%
- ROC AUC: 0.77

---

### 3. Simple Property-Only Estimation Model
A single-variable model provides quick estimates:

- R²: 21.2%
- RMSE: $9,019

This model is suitable for early-stage, high-level loan discussions.

---

### 4. Best Model: Four-Factor Regression
The optimal balance of accuracy and simplicity is achieved using:
- Property value
- Debt-to-income ratio
- Years on job
- Existing mortgage

**Performance:**
- R²: 24.0% (Adj. R²: 23.7%)
- RMSE: $8,860
- All predictors statistically significant (p < 0.01)

> Note: Mortgage and property value exhibit high multicollinearity (VIF > 15). Coefficients should not be interpreted independently, though predictive accuracy remains reliable.

---

### 5. Stepwise Validation
Automated stepwise selection independently chose the same four predictors, confirming model robustness.

---

## Model Comparison

| Model | R² | Adj. R² | RMSE | Predictors |
|------|----|--------|------|-----------|
| Simple Linear | 21.2% | 21.2% | $9,019 | 1 |
| Multiple Linear | 24.0% | 23.7% | $8,860 | 4 |
| Stepwise | 24.0% | 23.7% | $8,884 | 4 |

---

## Key Insights
- Property value dominates loan demand behavior
- Self-employed borrowers represent a high-value opportunity segment
- Approximately 76% of loan variation remains unexplained, likely due to missing variables such as income, credit scores, and loan purpose
- Models should guide—not replace—loan officer judgment

---

## Recommendations
- Deploy the four-factor model as a decision-support tool
- Pilot implementation in a single region
- Develop tailored products for self-employed borrowers
- De-emphasize low-impact variables such as credit inquiries
- Maintain human oversight in final lending decisions

---

## Conclusion
This analysis demonstrates that property value is the most influential determinant of loan demand. A practical four-factor predictive model provides meaningful improvements over simpler approaches while remaining suitable for operational use. These insights enable more proactive, data-driven lending strategies without replacing professional judgment.

---

## Appendix

### Logistic Regression Odds Ratios (High-Value Loans)

| Variable | Odds Ratio |
|--------|------------|
| Debt-to-Income | 1.02 |
| Years on Job | 1.05 |
| Job: Self-Employed | 2.42 |
| Job: Sales | 0.57 |
| Job: Other | 0.64 |

### Multicollinearity Diagnostics (VIF)

| Variable | VIF |
|--------|-----|
| Property Value | 17.6 |
| Mortgage | 15.2 |
| Debt-to-Income | 4.6 |
| Years on Job | 2.2 |


```python
# =============================================================================
# 1. Setup: Import Libraries and Load Data
# =============================================================================
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    df = pd.read_csv('Loan_data.csv')
except FileNotFoundError:
    print("Error: 'Loan_data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# =============================================================================
# 2. Data Cleaning and Preparation
# =============================================================================
print("--- Initial Data Information ---")
print(df.info())
print("\n--- Checking for Missing Values ---")
print(df.isnull().sum())

# Handle missing 'Job' values by filling with the mode ('Other')
# This is a common strategy for categorical variables.
job_mode = df['Job'].mode()[0]
df['Job'].fillna(job_mode, inplace=True)
print(f"\nMissing 'Job' values filled with '{job_mode}'.")

# Handle missing 'Loan' values by filling with the mean
loan_mean = df['Loan'].mean()
df['Loan'].fillna(loan_mean, inplace=True)
print(f"\nMissing 'Loan' values filled with '{loan_mean:.2f}'.")


# The CSV column names are slightly different from the data dictionary.
# Let's rename them for consistency with the assignment.
df.rename(columns={
    'Delinquencies': 'DELINQ',
    'Derogatories': 'DEROG',
    'Mortgage': 'MORTDUE',
    'Inquiries': 'NINQ',
    'Value': 'VALUE',
    'Loan': 'LOAN'
}, inplace=True)

# Define interval variables as per the data dictionary
interval_vars = ['CLAGE', 'CLNO', 'DEBTINC', 'LOAN', 'MORTDUE', 'NINQ', 'VALUE', 'YOJ']

print("\n--- Data successfully loaded and preprocessed ---")


# =============================================================================
# Question 1: Correlation Analysis
# =============================================================================
print("\n\n--- Question 1: Correlation of Interval Variables with Loan Amount ---")

# Calculate the correlation matrix for interval variables
correlation_matrix = df[interval_vars].corr()

# Extract the correlations specifically with the 'LOAN' variable
loan_correlations = correlation_matrix['LOAN'].sort_values(ascending=False)

print("Correlation of each interval variable with 'LOAN':")
print(loan_correlations)

# Identify the strongest and weakest related variables (excluding LOAN itself)
strongest_corr = loan_correlations[1:2]
weakest_corr = loan_correlations[-1:]
print(f"\nStrongest related variable: {strongest_corr.index[0]} (Correlation: {strongest_corr.values[0]:.2f})")
print(f"Weakest related variable: {weakest_corr.index[0]} (Correlation: {weakest_corr.values[0]:.2f})")

# Optional: Visualize the correlation matrix with a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Interval Variables')
plt.show()


# =============================================================================
# Question 2: Logistic Regression for High-Value Loans
# =============================================================================
print("\n\n--- Question 2: Logistic Regression to Predict High-Value Loans ---")

# a) Create the binary variable 'HighLoan'
df['HighLoan'] = (df['LOAN'] > 20000).astype(int)
print("Created 'HighLoan' variable:")
print(df['HighLoan'].value_counts())

# b) Build the Logistic Regression model
# Define predictors (X) and target (y)
predictors_log = ['DEBTINC', 'VALUE', 'YOJ', 'Job']
X = df[predictors_log]
y = df['HighLoan']

# 'Job' is categorical, so we need to create dummy variables
X = pd.get_dummies(X, columns=['Job'], drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Fit the logistic regression model using scikit-learn
log_model = LogisticRegression(solver='liblinear', random_state=42)
log_model.fit(X_train, y_train)

# c) Interpret Odds Ratios and Evaluate Accuracy
# Get coefficients to calculate odds ratios
odds_ratios = pd.DataFrame(np.exp(log_model.coef_[0]), index=X.columns, columns=['Odds Ratio'])
print("\nOdds Ratios for Predictors:")
print(odds_ratios)
print("\nInterpretation: An odds ratio > 1 means the predictor increases the odds of a HighLoan.")
print("For example, for every 1-unit increase in VALUE, the odds of having a HighLoan are multiplied by {:.4f}.".format(odds_ratios.loc['VALUE'][0]))


# Evaluate the model on the test set
y_pred = log_model.predict(X_test)

print("\n--- Model Evaluation ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate ROC AUC
y_pred_proba = log_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for HighLoan Prediction')
plt.legend(loc='lower right')
plt.show()


# =============================================================================
# Question 3: Simple Linear Regression
# =============================================================================
print("\n\n--- Question 3: Simple Linear Regression with the Strongest Predictor ---")

# The strongest predictor from Q1 was 'VALUE'
X_simple = df['VALUE']
y_linear = df['LOAN']

# Add a constant (intercept) to the predictor
X_simple_const = sm.add_constant(X_simple)

# Build and fit the model
simple_model = sm.OLS(y_linear, X_simple_const).fit()

# a) Interpret slope, intercept, and R-squared
print(simple_model.summary())
print("\n--- Interpretation of Simple Linear Regression ---")
intercept = simple_model.params['const']
slope = simple_model.params['VALUE']
r_squared = simple_model.rsquared

print(f"Intercept (const): {intercept:.2f}")
print("This is the predicted loan amount when property value is $0.")
print(f"Slope (VALUE): {slope:.4f}")
print("For every $1 increase in property value, the loan amount is predicted to increase by ${:.2f}.".format(slope))
print(f"R-squared: {r_squared:.4f}")
print(f"This model explains {r_squared:.1%} of the variability in the loan amount.")


# =============================================================================
# Question 4: Multiple Linear Regression
# =============================================================================
print("\n\n--- Question 4: Multiple Linear Regression ---")

# Define predictors and target
predictors_multi = ['DEBTINC', 'MORTDUE', 'VALUE', 'YOJ']
X_multi = df[predictors_multi]
y_multi = df['LOAN']

# Add a constant (intercept)
X_multi_const = sm.add_constant(X_multi)

# Build and fit the model
multi_model = sm.OLS(y_multi, X_multi_const).fit()

print(multi_model.summary())

# a) Check for multicollinearity using VIF
print("\n--- Multicollinearity Check (VIF) ---")
vif_data = pd.DataFrame()
vif_data["Variable"] = X_multi_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_multi_const.values, i) for i in range(X_multi_const.shape[1])]
print(vif_data)
print("\nNote: VIF > 5 suggests potential multicollinearity. 'VALUE' and 'MORTDUE' are highly correlated.")

print("\n--- Interpretation of Multiple Linear Regression ---")
print("Adjusted R-squared:", f"{multi_model.rsquared_adj:.4f}")
print(f"This model explains {multi_model.rsquared_adj:.1%} of the loan amount's variability, adjusting for the number of predictors.")
print("\nCoefficients:")
print(multi_model.params)
print("\nExample Interpretation (DEBTINC): For a one-unit increase in the debt-to-income ratio, the loan amount is predicted to increase by ${:.2f}, holding other variables constant.".format(multi_model.params['DEBTINC']))


# =============================================================================
# Question 5: Stepwise Regression (Forward Selection)
# =============================================================================
print("\n\n--- Question 5: Stepwise Regression (Forward Selection) ---")

def forward_selection(X, y, significance_level=0.05):
    """
    Performs forward selection for a linear regression model.
    X should be a pandas DataFrame of predictors.
    y should be a pandas Series of the target variable.
    """
    initial_features = X.columns.tolist()
    best_features = []

    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype=float)

        for new_column in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]

        min_p_value = new_pval.min()

        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break

    return best_features

# Prepare data for stepwise regression (use all potential interval and nominal predictors)
# We need to handle nominal variables by creating dummies first
df_dummies = pd.get_dummies(df.drop(columns=['ID']), drop_first=True)

# Convert boolean columns to integers
for col in df_dummies.columns:
    if df_dummies[col].dtype == 'bool':
        df_dummies[col] = df_dummies[col].astype(int)

# Define X and y for stepwise selection
X_stepwise = df_dummies.drop(columns=['LOAN', 'HighLoan'])
y_stepwise = df_dummies['LOAN']

# Perform forward selection
selected_features = forward_selection(X_stepwise, y_stepwise)
print("\nSelected features from stepwise regression:")
print(selected_features)

# Build the final stepwise model with selected features
X_stepwise_const = sm.add_constant(X_stepwise[selected_features])
stepwise_model = sm.OLS(y_stepwise, X_stepwise_const).fit()

print("\n--- Stepwise Regression Model Summary ---")
print(stepwise_model.summary())


# =============================================================================
# Question 6: Model Comparison
# =============================================================================
print("\n\n--- Question 6: Model Comparison ---")

# Compare R-squared and RMSE for the models
# Simple Model
simple_predictions = simple_model.predict(X_simple_const)
simple_rmse = np.sqrt(np.mean((y_linear - simple_predictions) ** 2))

# Multiple Model
multi_predictions = multi_model.predict(X_multi_const)
multi_rmse = np.sqrt(np.mean((y_multi - multi_predictions) ** 2))

# Stepwise Model
stepwise_predictions = stepwise_model.predict(X_stepwise_const)
stepwise_rmse = np.sqrt(np.mean((y_stepwise - stepwise_predictions) ** 2))

print("\nModel Comparison Summary:")
comparison_df = pd.DataFrame({
    'Model': ['Simple Linear', 'Multiple Linear', 'Stepwise'],
    'R-squared': [simple_model.rsquared, multi_model.rsquared, stepwise_model.rsquared],
    'Adjusted R-squared': [simple_model.rsquared_adj, multi_model.rsquared_adj, stepwise_model.rsquared_adj],
    'RMSE': [simple_rmse, multi_rmse, stepwise_rmse],
    'Number of Predictors': [1, len(predictors_multi), len(selected_features)]
})
print(comparison_df)
