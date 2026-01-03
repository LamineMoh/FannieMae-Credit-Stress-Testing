"""
================================================================================
PHASE 1: BUSINESS UNDERSTANDING
================================================================================
Fannie Mae 2008Q1 Stress Testing - Credit Default Risk Modeling
CRISP-DM Phase 1: Define Business Objectives and Success Criteria
================================================================================
"""

def display_business_context():
    """Display the business context and objectives for this analysis."""
    
    print("="*80)
    print("PHASE 1: BUSINESS UNDERSTANDING")
    print("="*80)
    
    business_context = """
    BUSINESS OBJECTIVE:
    -------------------
    • Predict mortgage loan defaults during Q1 2008 (financial crisis peak)
    • Support stress testing and risk assessment for credit portfolios
    • Identify key risk drivers to inform lending policies

    TARGET VARIABLE:
    ----------------
    • current_loan_delinquency: Loan delinquency status
      - 0 = Current (not delinquent) → No Default
      - 1-2 = 30-60 days delinquent → No Default  
      - 3+ = 90+ days delinquent → DEFAULT (Target)
      - RA = REO Acquisition → DEFAULT (Target)

    SUCCESS CRITERIA:
    -----------------
    • AUC-ROC > 0.70 (acceptable predictive power)
    • High Recall (minimize missed defaults - false negatives)
    • Interpretable feature importance for risk management
    
    STAKEHOLDERS:
    -------------
    • Risk Management Team
    • Credit Portfolio Managers
    • Regulatory Compliance Officers
    • Senior Leadership
    
    DELIVERABLES:
    -------------
    1. Trained classification model for default prediction
    2. Feature importance analysis
    3. Model performance report
    4. Recommendations for risk mitigation
    """
    
    print(business_context)
    return {
        'objective': 'Predict mortgage loan defaults during Q1 2008 crisis',
        'target_variable': 'is_default',
        'success_criteria': {'auc_roc': 0.70},
        'period': 'Q1 2008 (Financial Crisis Stress Period)'
    }


if __name__ == "__main__":
    business_config = display_business_context()
    print("\n" + "="*80)
    print("Business Understanding Complete!")
    print("="*80)
    print(f"\nConfiguration: {business_config}")
