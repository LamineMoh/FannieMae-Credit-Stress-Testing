"""
================================================================================
PHASE 4: MODELING
================================================================================
Fannie Mae 2008Q1 Stress Testing - Credit Default Risk Modeling
CRISP-DM Phase 4: Train Classification Models
================================================================================
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42


def initialize_models():
    """Initialize classification models."""
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
    return models


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Train a model and evaluate its performance."""
    print(f"\n>>> {model_name}")
    
    # Train model
    print("    Training...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1-Score:  {metrics['f1_score']:.4f}")
    print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    return metrics


def run_phase4(X_train, X_test, y_train, y_test):
    """Execute Phase 4: Modeling."""
    print("="*80)
    print("PHASE 4: MODELING")
    print("="*80)
    
    # Initialize models
    models = initialize_models()
    
    # Store results
    results = {}
    
    print("\nTraining and Evaluating Models...")
    print("-" * 60)
    
    for model_name, model in models.items():
        results[model_name] = train_and_evaluate(
            model, model_name, X_train, X_test, y_train, y_test
        )
    
    print("\n" + "="*80)
    print("Phase 4 Complete!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Import from previous phases
    from phase2_data_understanding import load_data
    from phase3_data_preparation import run_phase3
    
    df = load_data()
    prepared_data = run_phase3(df)
    
    results = run_phase4(
        prepared_data['X_train'],
        prepared_data['X_test'],
        prepared_data['y_train'],
        prepared_data['y_test']
    )
