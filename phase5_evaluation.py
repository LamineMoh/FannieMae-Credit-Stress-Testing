"""
================================================================================
PHASE 5: EVALUATION
================================================================================
Fannie Mae 2008Q1 Stress Testing - Credit Default Risk Modeling
CRISP-DM Phase 5: Evaluate and Compare Model Performance
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def compare_models(results):
    """Create model comparison summary."""
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
    return comparison_df


def select_best_model(comparison_df):
    """Select the best model based on AUC-ROC."""
    best_model_name = comparison_df['AUC-ROC'].idxmax()
    best_model_auc = comparison_df['AUC-ROC'].max()
    
    print(f"\n5.2 Best Model: {best_model_name} (AUC-ROC: {best_model_auc:.4f})")
    
    return best_model_name, best_model_auc


def analyze_confusion_matrix(y_test, y_pred, model_name):
    """Analyze confusion matrix for the model."""
    print(f"\n5.3 Confusion Matrix - {model_name}:")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"    True Negatives:  {cm[0,0]:,}")
    print(f"    False Positives: {cm[0,1]:,}")
    print(f"    False Negatives: {cm[1,0]:,}")
    print(f"    True Positives:  {cm[1,1]:,}")
    
    return cm


def show_classification_report(y_test, y_pred, model_name):
    """Display classification report."""
    print(f"\n5.4 Classification Report - {model_name}:")
    print(classification_report(y_test, y_pred, 
                              target_names=['No Default', 'Default']))


def analyze_feature_importance(model, features, model_name):
    """Analyze feature importance."""
    print("\n5.5 Feature Importance:")
    
    feature_importance_df = None
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        print(feature_importance_df.to_string(index=False))
        
    elif hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefs,
            'Abs_Importance': np.abs(coefs)
        }).sort_values('Abs_Importance', ascending=False)
        print(feature_importance_df.to_string(index=False))
    
    return feature_importance_df


def run_phase5(results, y_test, features):
    """Execute Phase 5: Evaluation."""
    print("="*80)
    print("PHASE 5: EVALUATION")
    print("="*80)
    
    # Compare models
    comparison_df = compare_models(results)
    
    # Select best model
    best_model_name, best_model_auc = select_best_model(comparison_df)
    best_results = results[best_model_name]
    
    # Confusion matrix
    cm = analyze_confusion_matrix(y_test, best_results['y_pred'], best_model_name)
    
    # Classification report
    show_classification_report(y_test, best_results['y_pred'], best_model_name)
    
    # Feature importance
    feature_importance_df = analyze_feature_importance(
        best_results['model'], features, best_model_name
    )
    
    print("\n" + "="*80)
    print("Phase 5 Complete!")
    print("="*80)
    
    return {
        'comparison_df': comparison_df,
        'best_model_name': best_model_name,
        'best_model_auc': best_model_auc,
        'confusion_matrix': cm,
        'feature_importance_df': feature_importance_df
    }


if __name__ == "__main__":
    # Import from previous phases
    from phase2_data_understanding import load_data
    from phase3_data_preparation import run_phase3
    from phase4_modeling import run_phase4
    
    df = load_data()
    prepared_data = run_phase3(df)
    
    results = run_phase4(
        prepared_data['X_train'],
        prepared_data['X_test'],
        prepared_data['y_train'],
        prepared_data['y_test']
    )
    
    evaluation = run_phase5(
        results, 
        prepared_data['y_test'],
        prepared_data['features']
    )
