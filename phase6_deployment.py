"""
================================================================================
PHASE 6: DEPLOYMENT
================================================================================
Fannie Mae 2008Q1 Stress Testing - Credit Default Risk Modeling
CRISP-DM Phase 6: Generate Visualizations and Reports
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def create_visualizations(results, y_test, cm, feature_importance_df, best_model_name):
    """Create comprehensive visualization dashboard."""
    print("\nCreating Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 6.1 ROC Curves
    ax1 = axes[0, 0]
    for model_name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
        ax1.plot(fpr, tpr, label=f"{model_name} (AUC={res['auc_roc']:.3f})", linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 6.2 Model Metrics Comparison
    ax2 = axes[0, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, (model_name, res) in enumerate(results.items()):
        values = [res['accuracy'], res['precision'], res['recall'], res['f1_score'], res['auc_roc']]
        ax2.bar(x + i*width, values, width, label=model_name, alpha=0.8, color=colors[i])
    
    ax2.set_xlabel('Metrics', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(metrics, rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 6.3 Confusion Matrix Heatmap
    ax3 = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'],
                annot_kws={'size': 14})
    ax3.set_xlabel('Predicted', fontsize=12)
    ax3.set_ylabel('Actual', fontsize=12)
    ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    
    # 6.4 Feature Importance
    ax4 = axes[1, 1]
    if feature_importance_df is not None:
        if 'Importance' in feature_importance_df.columns:
            ax4.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='steelblue')
        else:
            ax4.barh(feature_importance_df['Feature'], feature_importance_df['Abs_Importance'], color='steelblue')
        ax4.set_xlabel('Importance', fontsize=12)
        ax4.set_title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('2008Q1_CRISP_DM_Results.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved: 2008Q1_CRISP_DM_Results.png")
    
    return fig


def save_model_comparison(comparison_df):
    """Save model comparison to CSV."""
    comparison_df.to_csv('2008Q1_Model_Comparison.csv')
    print("✓ Model comparison saved: 2008Q1_Model_Comparison.csv")


def generate_summary_report(df, df_model, y, features, results, 
                           best_model_name, best_model_auc, cm, 
                           feature_importance_df):
    """Generate and save summary report."""
    
    feature_importance_str = feature_importance_df.to_string(index=False) if feature_importance_df is not None else 'N/A'
    
    summary_report = f"""
FANNIE MAE 2008Q1 STRESS TESTING - CREDIT DEFAULT PREDICTION
=============================================================

DATA OVERVIEW:
--------------
• Period: Q1 2008 (Financial Crisis Stress Period)
• Sample Size: {len(df):,} records analyzed
• Features Used: {', '.join(features)}
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
    
    with open('2008Q1_CRISP_DM_Report.txt', 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print("✓ Summary report saved: 2008Q1_CRISP_DM_Report.txt")
    
    return summary_report


def run_phase6(df, df_model, y, features, results, evaluation):
    """Execute Phase 6: Deployment."""
    print("="*80)
    print("PHASE 6: DEPLOYMENT")
    print("="*80)
    
    # Create visualizations
    create_visualizations(
        results,
        evaluation['y_test'] if 'y_test' in evaluation else None,
        evaluation['confusion_matrix'],
        evaluation['feature_importance_df'],
        evaluation['best_model_name']
    )
    
    # Save model comparison
    save_model_comparison(evaluation['comparison_df'])
    
    # Generate report
    summary_report = generate_summary_report(
        df, df_model, y, features, results,
        evaluation['best_model_name'],
        evaluation['best_model_auc'],
        evaluation['confusion_matrix'],
        evaluation['feature_importance_df']
    )
    
    print("\n" + "="*80)
    print("CRISP-DM ANALYSIS COMPLETE!")
    print("="*80)
    
    return summary_report


if __name__ == "__main__":
    # Import from previous phases
    from phase2_data_understanding import load_data
    from phase3_data_preparation import run_phase3
    from phase4_modeling import run_phase4
    from phase5_evaluation import run_phase5
    
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
    evaluation['y_test'] = prepared_data['y_test']
    
    y = prepared_data['df_model']['is_default'].values.astype(int)
    
    summary = run_phase6(
        df,
        prepared_data['df_model'],
        y,
        prepared_data['features'],
        results,
        evaluation
    )
