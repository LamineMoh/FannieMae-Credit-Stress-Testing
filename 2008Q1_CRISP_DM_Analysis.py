"""
================================================================================
CRISP-DM ANALYSIS: FANNIE MAE 2008Q1 STRESS TESTING
================================================================================
Author: Data Science Team
Date: 2025-12-30
Description: Complete CRISP-DM workflow for credit default risk modeling
             during the 2008 financial crisis stress period
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score
)

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = r"c:\Users\zouta\Desktop\Performance_All\stress_period_2007_2009\2008Q1.csv"
OUTPUT_DIR = r"c:\Users\zouta\Desktop\Performance_All\stress_period_2007_2009"
SAMPLE_SIZE = 500000  # Sample size for analysis (adjust based on memory)
RANDOM_STATE = 42

# Fannie Mae Performance File Columns (110 columns)
# Based on Fannie Mae Single-Family Loan Performance Data documentation
COLUMN_NAMES = [
    'loan_sequence_number',         # 0 - Loan Identifier
    'monthly_reporting_period',     # 1 - Monthly Reporting Period (YYYYMM)
    'current_actual_upb',           # 2 - Current Actual UPB
    'current_loan_delinquency',     # 3 - Current Loan Delinquency Status (TARGET)
    'loan_age',                     # 4 - Loan Age
    'remaining_months_maturity',    # 5 - Remaining Months to Legal Maturity
    'repurchase_flag',              # 6 - Repurchase Flag
    'modification_flag',            # 7 - Modification Flag
    'zero_balance_code',            # 8 - Zero Balance Code
    'zero_balance_date',            # 9 - Zero Balance Effective Date
    'current_interest_rate',        # 10 - Current Interest Rate
    'current_deferred_upb',         # 11 - Current Deferred UPB
    'due_date_last_paid',           # 12 - Due Date of Last Paid Installment
    'mi_recoveries',                # 13 - MI Recoveries
    'net_sales_proceeds',           # 14 - Net Sales Proceeds
    'non_mi_recoveries',            # 15 - Non MI Recoveries
    'expenses',                     # 16 - Expenses
    'legal_costs',                  # 17 - Legal Costs  
    'maintenance_costs',            # 18 - Maintenance and Preservation Costs
    'taxes_insurance_due',          # 19 - Taxes and Insurance
    'miscellaneous_expenses',       # 20 - Miscellaneous Expenses
    'actual_loss_calculation',      # 21 - Actual Loss Calculation
    'modification_cost',            # 22 - Modification Cost
    'step_modification_flag',       # 23 - Step Modification Flag
    'deferred_payment_mod',         # 24 - Deferred Payment Plan
    'estimated_ltv',                # 25 - Estimated Loan-to-Value
    'zero_balance_removal_upb',     # 26 - Zero Balance Removal UPB
    'delinquent_accrued_interest',  # 27 - Delinquent Accrued Interest
    'delinquency_due_disaster',     # 28 - Delinquency Due to Disaster
    'borrower_assistance_status',   # 29 - Borrower Assistance Status Code
]

# Add remaining columns as generic names
for i in range(30, 110):
    COLUMN_NAMES.append(f'col_{i}')

print("="*80)
print("CRISP-DM ANALYSIS: FANNIE MAE 2008Q1 STRESS TESTING")
print("="*80)
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# =============================================================================
# PHASE 1: BUSINESS UNDERSTANDING
# =============================================================================
print("\n" + "="*80)
print("PHASE 1: BUSINESS UNDERSTANDING")
print("="*80)

business_context = """
BUSINESS OBJECTIVE:
-------------------
- Predict mortgage loan defaults during Q1 2008 (peak of financial crisis)
- Support stress testing and risk assessment for credit portfolios
- Identify key risk drivers to inform lending policies

TARGET VARIABLE:
----------------
- current_loan_delinquency: Loan delinquency status
  * 0 = Current (not delinquent)
  * 1-2 = 30-60 days delinquent
  * 3+ = 90+ days delinquent (DEFAULT - our target)
  * RA = REO Acquisition

SUCCESS CRITERIA:
-----------------
- AUC-ROC > 0.70 (acceptable predictive power)
- High Recall (minimize missed defaults - false negatives)
- Interpretable feature importance for risk management
"""
print(business_context)

# =============================================================================
# PHASE 2: DATA UNDERSTANDING
# =============================================================================
print("\n" + "="*80)
print("PHASE 2: DATA UNDERSTANDING")
print("="*80)

# 2.1 Load Data Sample
print("\n2.1 Loading Data Sample...")
print(f"    Reading sample of {SAMPLE_SIZE:,} rows from 2008Q1.csv...")

try:
    # Read with proper settings for Fannie Mae data
    df = pd.read_csv(
        DATA_PATH,
        sep='|',
        header=None,
        names=COLUMN_NAMES,
        nrows=SAMPLE_SIZE,
        low_memory=False,
        on_bad_lines='skip'
    )
    print(f"    ✓ Successfully loaded {len(df):,} records with {df.shape[1]} columns")
except Exception as e:
    print(f"    ✗ Error loading data: {e}")
    raise

# 2.2 Data Structure
print("\n2.2 Data Structure:")
print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"    Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2.3 Identify columns with data
print("\n2.3 Analyzing Column Data Availability:")
col_data_counts = {}
for col in df.columns:
    non_null_count = df[col].notna().sum()
    if non_null_count > 0:
        col_data_counts[col] = non_null_count

print(f"    Columns with data: {len(col_data_counts)} out of {len(df.columns)}")
print("\n    Top 15 columns with most data:")
sorted_cols = sorted(col_data_counts.items(), key=lambda x: x[1], reverse=True)[:15]
for col, count in sorted_cols:
    pct = count / len(df) * 100
    print(f"    - {col}: {count:,} records ({pct:.1f}%)")

# 2.4 Sample of key columns
print("\n2.4 Sample Data (First 10 rows, Key Columns):")
key_cols = ['loan_sequence_number', 'monthly_reporting_period', 'current_actual_upb', 
            'current_loan_delinquency', 'loan_age', 'current_interest_rate', 'estimated_ltv']
available_key_cols = [c for c in key_cols if c in df.columns and df[c].notna().sum() > 0]
if available_key_cols:
    print(df[available_key_cols].head(10).to_string())
else:
    print("    No key columns found with data. Showing first 10 columns:")
    print(df.iloc[:10, :10].to_string())

# 2.5 Target Variable Analysis
print("\n2.5 Target Variable Distribution (current_loan_delinquency):")
if 'current_loan_delinquency' in df.columns:
    delinq_dist = df['current_loan_delinquency'].value_counts(dropna=False)
    print(delinq_dist.head(20).to_string())
else:
    # Try to find delinquency column
    print("    Looking for delinquency status in data...")
    for i, col in enumerate(df.columns):
        sample_values = df[col].dropna().head(20).unique()
        if any(str(v) in ['0', '1', '2', '3', 'RA', 'XX'] for v in sample_values):
            print(f"    Potential delinquency column {i} ({col}):")
            print(f"    Values: {df[col].value_counts(dropna=False).head(10).to_string()}")
            break

# =============================================================================
# PHASE 3: DATA PREPARATION  
# =============================================================================
print("\n" + "="*80)
print("PHASE 3: DATA PREPARATION")
print("="*80)

# 3.1 Identify numeric features with data
print("\n3.1 Identifying Usable Features...")

# Find columns with actual numeric data
numeric_features = []
for col in df.columns:
    # Try to convert to numeric
    numeric_vals = pd.to_numeric(df[col], errors='coerce')
    valid_count = numeric_vals.notna().sum()
    if valid_count > len(df) * 0.5:  # At least 50% valid
        # Check if it has reasonable variance
        if numeric_vals.std() > 0:
            numeric_features.append(col)
            
print(f"    Found {len(numeric_features)} columns with usable numeric data:")
for col in numeric_features[:10]:
    print(f"    - {col}")

# 3.2 Create Target Variable
print("\n3.2 Creating Target Variable...")

# Find and use the delinquency column (column index 3 typically)
target_col = 'current_loan_delinquency'

def create_default_flag(status):
    """
    Create binary default indicator:
    - 0: No default (current or minor delinquency)
    - 1: Default (90+ days delinquent or foreclosure)
    """
    if pd.isna(status):
        return np.nan
    status_str = str(status).strip().upper()
    
    # Default conditions
    if status_str in ['RA', 'XX', 'F', 'R', 'S', 'T', 'N', '09', '06']:
        return 1
    try:
        if int(float(status_str)) >= 3:  # 90+ days delinquent
            return 1
    except:
        pass
    return 0

if target_col in df.columns:
    df['is_default'] = df[target_col].apply(create_default_flag)
else:
    # Use column index 3 if column name not found
    print("    Using column index 3 as delinquency status...")
    df['is_default'] = df.iloc[:, 3].apply(create_default_flag)

default_dist = df['is_default'].value_counts(dropna=False)
print("    Target Distribution:")
print(f"    - No Default (0): {default_dist.get(0, 0):,} ({default_dist.get(0, 0)/len(df)*100:.2f}%)")
print(f"    - Default (1): {default_dist.get(1, 0):,} ({default_dist.get(1, 0)/len(df)*100:.2f}%)")
print(f"    - Missing: {df['is_default'].isna().sum():,}")

# 3.3 Select features for modeling
print("\n3.3 Selecting Features for Modeling...")

# Use the first few numeric features that have good data coverage
# Common useful features: UPB, loan_age, interest_rate, LTV
feature_candidates = [
    'current_actual_upb', 'loan_age', 'remaining_months_maturity',
    'current_interest_rate', 'estimated_ltv', 'current_deferred_upb'
]

# Also try column indices if column names don't work
available_features = []
for col in feature_candidates:
    if col in df.columns:
        numeric_vals = pd.to_numeric(df[col], errors='coerce')
        valid_count = numeric_vals.notna().sum()
        if valid_count > len(df) * 0.3:  # At least 30% valid
            available_features.append(col)

# If no named features work, use positional columns
if len(available_features) < 2:
    print("    Named features not found. Using positional columns...")
    for i in [2, 4, 5, 10, 25]:  # Common indices for UPB, loan_age, remaining_months, interest_rate, LTV
        if i < len(df.columns):
            col = df.columns[i]
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            valid_count = numeric_vals.notna().sum()
            if valid_count > len(df) * 0.3:
                available_features.append(col)
                
print(f"    Selected features: {available_features}")

# 3.4 Prepare modeling dataset
print("\n3.4 Preparing Modeling Dataset...")

# Create modeling dataframe
df_model = df[available_features + ['is_default']].copy()

# Convert all features to numeric
for col in available_features:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    valid_count = df_model[col].notna().sum()
    print(f"    - {col}: {valid_count:,} valid numeric values")

# 3.5 Handle Missing Values
print("\n3.5 Handling Missing Values...")

# Remove rows with missing target
df_model = df_model.dropna(subset=['is_default'])
print(f"    After removing missing targets: {len(df_model):,} records")

# Fill missing features with median
for col in available_features:
    missing_count = df_model[col].isnull().sum()
    if missing_count > 0:
        median_val = df_model[col].median()
        if pd.notna(median_val):
            df_model[col].fillna(median_val, inplace=True)
            print(f"    - Filled {col}: {missing_count:,} missing with median {median_val:.2f}")

# 3.6 Remove outliers and invalid values
print("\n3.6 Removing Outliers and Invalid Values...")
initial_count = len(df_model)

# Remove infinite values
df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

# Remove extreme outliers (1st and 99th percentile)
for col in available_features:
    if df_model[col].dtype in ['int64', 'float64'] and len(df_model) > 0:
        Q1 = df_model[col].quantile(0.01)
        Q99 = df_model[col].quantile(0.99)
        df_model = df_model[(df_model[col] >= Q1) & (df_model[col] <= Q99)]

print(f"    Removed {initial_count - len(df_model):,} outlier/invalid records")
print(f"    Final dataset size: {len(df_model):,} records")

# Check if we have enough data
if len(df_model) < 100:
    print("\n    ERROR: Insufficient data after cleaning!")
    print("    Attempting alternative approach with less strict cleaning...")
    
    # Retry with less strict cleaning
    df_model = df[available_features + ['is_default']].copy()
    for col in available_features:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    df_model = df_model.dropna()
    print(f"    Retry dataset size: {len(df_model):,} records")
    
    if len(df_model) < 100:
        raise ValueError(f"Insufficient data: only {len(df_model)} valid records found")

# 3.7 Feature Scaling
print("\n3.7 Feature Scaling (StandardScaler)...")
scaler = StandardScaler()
X = df_model[available_features].values
y = df_model['is_default'].values.astype(int)

X_scaled = scaler.fit_transform(X)
print("    ✓ Features standardized (mean=0, std=1)")

# 3.8 Train-Test Split
print("\n3.8 Train-Test Split (80/20)...")

# Check class distribution for stratification
unique_classes = np.unique(y)
print(f"    Classes: {unique_classes}")
print(f"    Class distribution: {np.bincount(y)}")

# Use stratify only if both classes are present
if len(unique_classes) > 1 and min(np.bincount(y)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )

print(f"    Training set: {len(X_train):,} samples")
print(f"    Test set: {len(X_test):,} samples")
print(f"    Default rate (train): {y_train.mean()*100:.2f}%")
print(f"    Default rate (test): {y_test.mean()*100:.2f}%")

# =============================================================================
# PHASE 4: MODELING
# =============================================================================
print("\n" + "="*80)
print("PHASE 4: MODELING")
print("="*80)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(
        random_state=RANDOM_STATE, 
        max_iter=1000,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE
    )
}

# Store results
results = {}

print("\nTraining and Evaluating Models...")
print("-" * 60)

for model_name, model in models.items():
    print(f"\n>>> {model_name}")
    
    # Train model
    print("    Training...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    AUC-ROC:   {auc_roc:.4f}")

# =============================================================================
# PHASE 5: EVALUATION
# =============================================================================
print("\n" + "="*80)
print("PHASE 5: EVALUATION")
print("="*80)

# 5.1 Model Comparison Summary
print("\n5.1 Model Comparison Summary:")
print("-" * 80)

comparison_df = pd.DataFrame({
    model_name: {
        'Accuracy': res['accuracy'],
        'Precision': res['precision'],
        'Recall': res['recall'],
        'F1-Score': res['f1_score'],
        'AUC-ROC': res['auc_roc']
    }
    for model_name, res in results.items()
}).T

print(comparison_df.round(4).to_string())

# 5.2 Best Model Selection
best_model_name = comparison_df['AUC-ROC'].idxmax()
best_model_auc = comparison_df['AUC-ROC'].max()
print(f"\n5.2 Best Model: {best_model_name} (AUC-ROC: {best_model_auc:.4f})")

# 5.3 Confusion Matrix for Best Model
print(f"\n5.3 Confusion Matrix - {best_model_name}:")
best_results = results[best_model_name]
cm = confusion_matrix(y_test, best_results['y_pred'])
print(f"    True Negatives:  {cm[0,0]:,}")
print(f"    False Positives: {cm[0,1]:,}")
print(f"    False Negatives: {cm[1,0]:,}")
print(f"    True Positives:  {cm[1,1]:,}")

# 5.4 Classification Report
print(f"\n5.4 Classification Report - {best_model_name}:")
print(classification_report(y_test, best_results['y_pred'], 
                          target_names=['No Default', 'Default']))

# 5.5 Feature Importance (if available)
print("\n5.5 Feature Importance:")
feature_importance_df = None
if hasattr(results[best_model_name]['model'], 'feature_importances_'):
    importances = results[best_model_name]['model'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    print(feature_importance_df.to_string(index=False))
elif hasattr(results[best_model_name]['model'], 'coef_'):
    coefs = results[best_model_name]['model'].coef_[0]
    feature_importance_df = pd.DataFrame({
        'feature': available_features,
        'coefficient': coefs,
        'abs_importance': np.abs(coefs)
    }).sort_values('abs_importance', ascending=False)
    print(feature_importance_df.to_string(index=False))

# =============================================================================
# PHASE 6: VISUALIZATION & REPORTING
# =============================================================================
print("\n" + "="*80)
print("PHASE 6: VISUALIZATION & REPORTING")
print("="*80)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 6.1 ROC Curves
ax1 = axes[0, 0]
for model_name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
    ax1.plot(fpr, tpr, label=f"{model_name} (AUC={res['auc_roc']:.3f})")
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves Comparison')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# 6.2 Model Metrics Comparison
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(metrics))
width = 0.25

for i, (model_name, res) in enumerate(results.items()):
    values = [res['accuracy'], res['precision'], res['recall'], res['f1_score'], res['auc_roc']]
    ax2.bar(x + i*width, values, width, label=model_name, alpha=0.8)

ax2.set_xlabel('Metrics')
ax2.set_ylabel('Score')
ax2.set_title('Model Performance Comparison')
ax2.set_xticks(x + width)
ax2.set_xticklabels(metrics, rotation=45)
ax2.legend()
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3, axis='y')

# 6.3 Confusion Matrix Heatmap
ax3 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')
ax3.set_title(f'Confusion Matrix - {best_model_name}')

# 6.4 Feature Importance
ax4 = axes[1, 1]
if feature_importance_df is not None:
    if 'importance' in feature_importance_df.columns:
        ax4.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    else:
        ax4.barh(feature_importance_df['feature'], feature_importance_df['abs_importance'])
    ax4.set_xlabel('Importance')
    ax4.set_title(f'Feature Importance - {best_model_name}')
    ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()

# Save figure
output_path = os.path.join(OUTPUT_DIR, '2008Q1_CRISP_DM_Results.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n    ✓ Visualization saved: {output_path}")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS SUMMARY REPORT")
print("="*80)

feature_importance_str = feature_importance_df.to_string(index=False) if feature_importance_df is not None else 'N/A'

summary_report = f"""
FANNIE MAE 2008Q1 STRESS TESTING - CREDIT DEFAULT PREDICTION
=============================================================

DATA OVERVIEW:
--------------
• Period: Q1 2008 (Financial Crisis Stress Period)
• Sample Size: {len(df):,} records analyzed
• Features Used: {', '.join(available_features)}
• Target Variable: is_default (binary: 0=No Default, 1=Default)
• Default Rate: {y.mean()*100:.2f}%
• Clean Dataset Size: {len(df_model):,} records

BEST MODEL: {best_model_name}
---------------------------------
• AUC-ROC:   {best_model_auc:.4f}
• Accuracy:  {results[best_model_name]['accuracy']:.4f}
• Precision: {results[best_model_name]['precision']:.4f}
• Recall:    {results[best_model_name]['recall']:.4f}
• F1-Score:  {results[best_model_name]['f1_score']:.4f}

STRESS TEST INSIGHTS:
---------------------
1. Model achieves {'ACCEPTABLE' if best_model_auc >= 0.7 else 'BELOW TARGET'} 
   predictive power (AUC {'≥' if best_model_auc >= 0.7 else '<'} 0.70)
   
2. Key Risk Drivers (by importance):
   {feature_importance_str}

3. The model can identify {results[best_model_name]['recall']*100:.1f}% of actual defaults
   (Recall metric - critical for risk management)

CONFUSION MATRIX:
-----------------
True Negatives:  {cm[0,0]:,}
False Positives: {cm[0,1]:,}
False Negatives: {cm[1,0]:,}
True Positives:  {cm[1,1]:,}

RECOMMENDATIONS:
----------------
1. Monitor loans with high-risk feature values identified above
2. Implement early warning system for loans approaching delinquency
3. Consider enhancing model with additional features (FICO score, LTV, DTI)
4. Regular model recalibration with updated crisis data

Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

print(summary_report)

# Save summary report
report_path = os.path.join(OUTPUT_DIR, '2008Q1_CRISP_DM_Report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(summary_report)
print(f"\n✓ Summary report saved: {report_path}")

# Save model comparison to CSV
comparison_path = os.path.join(OUTPUT_DIR, '2008Q1_Model_Comparison.csv')
comparison_df.to_csv(comparison_path)
print(f"✓ Model comparison saved: {comparison_path}")

print("\n" + "="*80)
print("CRISP-DM ANALYSIS COMPLETE")
print("="*80)
