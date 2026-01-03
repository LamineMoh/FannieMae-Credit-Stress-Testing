"""
================================================================================
PHASE 2: DATA UNDERSTANDING
================================================================================
Fannie Mae 2008Q1 Stress Testing - Credit Default Risk Modeling
CRISP-DM Phase 2: Load, Explore, and Describe the Data
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = r"2008Q1.csv"
SAMPLE_SIZE = 500000

# Column Names for Fannie Mae Performance File
COLUMN_NAMES = [
    'loan_sequence_number',         
    'monthly_reporting_period',     
    'current_actual_upb',           
    'current_loan_delinquency',     
    'loan_age',                     
    'remaining_months_maturity',    
    'repurchase_flag',              
    'modification_flag',            
    'zero_balance_code',            
    'zero_balance_date',            
    'current_interest_rate',        
    'current_deferred_upb',         
    'due_date_last_paid',           
    'mi_recoveries',                
    'net_sales_proceeds',           
    'non_mi_recoveries',            
    'expenses',                     
    'legal_costs',                  
    'maintenance_costs',            
    'taxes_insurance_due',          
    'miscellaneous_expenses',       
    'actual_loss_calculation',      
    'modification_cost',            
    'step_modification_flag',       
    'deferred_payment_mod',         
    'estimated_ltv',                
    'zero_balance_removal_upb',     
    'delinquent_accrued_interest',  
    'delinquency_due_disaster',     
    'borrower_assistance_status',   
]

# Add remaining columns
for i in range(30, 110):
    COLUMN_NAMES.append(f'col_{i}')


def load_data(data_path=DATA_PATH, sample_size=SAMPLE_SIZE):
    """Load sample data from CSV file."""
    print(f"\n2.1 Loading Data Sample...")
    print(f"    Reading {sample_size:,} rows from {data_path}...")
    
    df = pd.read_csv(
        data_path,
        sep='|',
        header=None,
        names=COLUMN_NAMES,
        nrows=sample_size,
        low_memory=False,
        on_bad_lines='skip'
    )
    print(f"    ✓ Loaded {len(df):,} records with {df.shape[1]} columns")
    return df


def explore_structure(df):
    """Explore data structure."""
    print("\n2.2 Data Structure:")
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"    Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n    Column Types:")
    print(df.dtypes.value_counts())


def analyze_columns(df):
    """Analyze which columns have data."""
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
        print(f"    - {col}: {count:,} ({pct:.1f}%)")
    
    return col_data_counts


def analyze_target(df, target_col='current_loan_delinquency'):
    """Analyze target variable distribution."""
    print(f"\n2.4 Target Variable Distribution ({target_col}):")
    
    if target_col in df.columns:
        delinq_dist = df[target_col].value_counts(dropna=False)
        print(delinq_dist.head(20))
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 5))
        delinq_dist.head(10).plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title('Distribution of Loan Delinquency Status', fontsize=14, fontweight='bold')
        ax.set_xlabel('Delinquency Status')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('2008Q1_Target_Distribution.png', dpi=150)
        plt.show()
        print("    ✓ Saved: 2008Q1_Target_Distribution.png")
        
        return delinq_dist
    return None


def run_phase2():
    """Execute Phase 2: Data Understanding."""
    print("="*80)
    print("PHASE 2: DATA UNDERSTANDING")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Explore structure
    explore_structure(df)
    
    # Analyze columns
    col_data_counts = analyze_columns(df)
    
    # Analyze target
    target_dist = analyze_target(df)
    
    print("\n" + "="*80)
    print("Phase 2 Complete!")
    print("="*80)
    
    return df, col_data_counts, target_dist


if __name__ == "__main__":
    df, col_data_counts, target_dist = run_phase2()
