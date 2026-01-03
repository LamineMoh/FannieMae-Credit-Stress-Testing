"""
================================================================================
PHASE 3: DATA PREPARATION
================================================================================
Fannie Mae 2008Q1 Stress Testing - Credit Default Risk Modeling
CRISP-DM Phase 3: Clean, Transform, and Prepare Data for Modeling
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42


def create_default_flag(status):
    """
    Create binary default indicator.
    
    Parameters:
    -----------
    status : str or numeric
        Delinquency status value
        
    Returns:
    --------
    int : 0 for no default, 1 for default, NaN for missing
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


def create_target_variable(df, target_col='current_loan_delinquency'):
    """Create binary target variable is_default."""
    print("\n3.1 Creating Binary Target Variable (is_default)...")
    
    df['is_default'] = df[target_col].apply(create_default_flag)
    
    default_dist = df['is_default'].value_counts(dropna=False)
    print("    Target Distribution:")
    print(f"    - No Default (0): {default_dist.get(0, 0):,} ({default_dist.get(0, 0)/len(df)*100:.2f}%)")
    print(f"    - Default (1): {default_dist.get(1, 0):,} ({default_dist.get(1, 0)/len(df)*100:.2f}%)")
    print(f"    - Missing: {df['is_default'].isna().sum():,}")
    
    return df


def select_features(df):
    """Select features for modeling."""
    print("\n3.2 Selecting Features for Modeling...")
    
    feature_candidates = [
        'current_actual_upb', 'loan_age', 'remaining_months_maturity',
        'current_interest_rate', 'estimated_ltv', 'current_deferred_upb'
    ]
    
    available_features = []
    for col in feature_candidates:
        if col in df.columns:
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            valid_count = numeric_vals.notna().sum()
            if valid_count > len(df) * 0.3:
                available_features.append(col)
    
    # Fallback to positional columns
    if len(available_features) < 2:
        print("    Named features not found. Using positional columns...")
        for i in [2, 4, 5, 10, 25, 11]:
            if i < len(df.columns):
                col = df.columns[i]
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                if numeric_vals.notna().sum() > len(df) * 0.3:
                    available_features.append(col)
    
    print(f"    Selected features: {available_features}")
    return available_features


def convert_to_numeric(df, features):
    """Convert features to numeric values."""
    print("\n3.3 Converting Features to Numeric...")
    
    df_model = df[features + ['is_default']].copy()
    
    for col in features:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
        valid_count = df_model[col].notna().sum()
        print(f"    - {col}: {valid_count:,} valid numeric values")
    
    return df_model


def handle_missing_values(df_model, features):
    """Handle missing values in the dataset."""
    print("\n3.4 Handling Missing Values...")
    
    # Remove rows with missing target
    df_model = df_model.dropna(subset=['is_default'])
    print(f"    After removing missing targets: {len(df_model):,} records")
    
    # Fill missing features with median
    for col in features:
        missing_count = df_model[col].isnull().sum()
        if missing_count > 0:
            median_val = df_model[col].median()
            if pd.notna(median_val):
                df_model[col].fillna(median_val, inplace=True)
                print(f"    - Filled {col}: {missing_count:,} missing with median {median_val:.2f}")
    
    return df_model


def remove_outliers(df_model, features):
    """Remove outliers and invalid values."""
    print("\n3.5 Removing Outliers and Invalid Values...")
    initial_count = len(df_model)
    
    # Remove infinite values
    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Remove extreme outliers (1st and 99th percentile)
    for col in features:
        if df_model[col].dtype in ['int64', 'float64'] and len(df_model) > 0:
            Q1 = df_model[col].quantile(0.01)
            Q99 = df_model[col].quantile(0.99)
            df_model = df_model[(df_model[col] >= Q1) & (df_model[col] <= Q99)]
    
    print(f"    Removed {initial_count - len(df_model):,} outlier/invalid records")
    print(f"    Final dataset size: {len(df_model):,} records")
    
    return df_model


def scale_features(df_model, features):
    """Scale features using StandardScaler."""
    print("\n3.6 Feature Scaling (StandardScaler)...")
    
    scaler = StandardScaler()
    X = df_model[features].values
    y = df_model['is_default'].values.astype(int)
    
    X_scaled = scaler.fit_transform(X)
    print("    ✓ Features standardized (mean=0, std=1)")
    
    return X_scaled, y, scaler


def split_data(X_scaled, y, test_size=0.2):
    """Split data into train and test sets."""
    print("\n3.7 Train-Test Split (80/20)...")
    
    unique_classes = np.unique(y)
    print(f"    Classes: {unique_classes}")
    print(f"    Class distribution: {np.bincount(y)}")
    
    # Use stratify if both classes present
    if len(unique_classes) > 1 and min(np.bincount(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=RANDOM_STATE
        )
    
    print(f"    Training set: {len(X_train):,} samples")
    print(f"    Test set: {len(X_test):,} samples")
    print(f"    Default rate (train): {y_train.mean()*100:.2f}%")
    print(f"    Default rate (test): {y_test.mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def run_phase3(df):
    """Execute Phase 3: Data Preparation."""
    print("="*80)
    print("PHASE 3: DATA PREPARATION")
    print("="*80)
    
    # Create target variable
    df = create_target_variable(df)
    
    # Select features
    features = select_features(df)
    
    # Convert to numeric
    df_model = convert_to_numeric(df, features)
    
    # Handle missing values
    df_model = handle_missing_values(df_model, features)
    
    # Remove outliers
    df_model = remove_outliers(df_model, features)
    
    # Scale features
    X_scaled, y, scaler = scale_features(df_model, features)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    print("\n" + "="*80)
    print("Phase 3 Complete!")
    print("="*80)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': features,
        'scaler': scaler,
        'df_model': df_model
    }


if __name__ == "__main__":
    # Import from phase 2
    from phase2_data_understanding import load_data
    
    df = load_data()
    prepared_data = run_phase3(df)
