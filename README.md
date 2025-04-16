# Telecom Churn Prevention Email Generator

## Project Overview
This project combines ML-based churn prediction with LLM-generated personalized retention emails for telecom customers.

## Notebook Analysis Summary

| Component | Key Findings |
|-----------|--------------|
| **Data** | 7,043 customers, 21 features, 26.6% churn rate |
| **Top Churn Factors** | Month-to-month contracts (75% churn), high charges, fiber optic, no tech support |
| **Best Models** | Voting Classifier (81.6%), Random Forest (81.3%), Logistic Regression (80.9%) |

## Email Generator App Implementation

1. **UI Risk Toggles**: Based on ML-identified factors
   ```python
   # Toggles are derived directly from ML model insights
   create_toggle("monthToMonth", "Month-to-Month Contract", checked=True)  # Strongest churn predictor
   create_toggle("fiberOptic", "Fiber Optic")  # High churn signal
   ```

2. **Strategy Mapping**: Risk factors linked to targeted solutions
   ```python
   # Each toggle triggers specific retention strategies
   RISK_FACTOR_STRATEGIES = {
       "highMonthlyCharges": ["Discount on 1-year contract", "Loyalty price reduction"],
       "fiberOptic": ["Free speed upgrade", "Enhanced reliability promise"]
   }
   ```

3. **Dynamic Prompt Construction**: Assembles personalized LLM inputs
   ```python
   # Converts UI selections into structured prompt components
   if monthly_charges > 75:
       risk_factors.append(f"High monthly charges (${monthly_charges})")
       retention_strategies.extend(RISK_FACTOR_STRATEGIES["highMonthlyCharges"])
   ```

## Key Questions & Answers

### Customer Churn Prediction
1. **Classification Performance**
   - Best: Voting Classifier (81.6% accuracy)
   - Metrics: F1 scores of 0.88 (non-churn), 0.62 (churn)

2. **Influential Factors**
   - Contract type (month-to-month vs. annual)
   - Service combinations (fiber without security)
   - Demographics (single customers churn more)
   - Payment methods (electronic check highest risk)

### Email Generation
1. **Brand Guideline Adherence**
   - Tone instructions embedded in prompts
   - Structured format with standardized signature
   - Temperature (0.7) balances consistency with personalization

2. **Scaling Potential**
   - Lifecycle-based segmentation and triggers
   - Multi-channel extension
   - A/B testing and feedback integration

## Future Improvements
1. SHAP analysis for automatic risk identification
2. Batch processing via CSV upload
3. Multi-channel notifications based on preferences
4. Performance analytics by customer segment
5. Reinforcement learning for offer optimization
