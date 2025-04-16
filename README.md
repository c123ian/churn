# Telecom Churn Prevention Email Generator

![APP 1](https://github.com/user-attachments/assets/e30fbfa1-a76a-490f-9bda-9e0df2ab3b66)

![APP 2](https://github.com/user-attachments/assets/3b067dec-469a-4190-9d70-113aeca34a5f)


## 🔍 Live Applications
- **App 1: ML Churn Predictor** - [https://c123ian--churn-predictor-serve.modal.run/](https://c123ian--churn-predictor-serve.modal.run/)
- **App 2: LLM Email Generator** - [https://c123ian--email-generator-serve.modal.run/](https://c123ian--email-generator-serve.modal.run/)

## Project Structure & Workflow
This project combines ML-based churn prediction with LLM-generated personalized retention emails for telecom customers.

1. **`notebook4d371d4ce.ipynb`** - Data analysis and model selection
2. **`train_save_model.py`** - Trains and saves the selected ML algorithm (Voting Classifier)
3. **`churn_pred.py`** - **App 1**: ML-based churn risk prediction application
4. **`app.py`** - **App 2**: LLM-powered personalized email generator for retention

## Deployment Order
1. Train and save model: `modal run train_save_model.py`
2. Deploy prediction app: `modal deploy churn_pred.py`
3. Deploy email generator app: `modal deploy app.py`

## Notebook Analysis Summary

| Component | Key Findings |
|-----------|--------------|
| **Data** | 7,043 customers, 21 features, 26.6% churn rate |
| **Top Churn Factors** | Month-to-month contracts (75% churn), high charges, fiber optic, no tech support |
| **Best Models** | Voting Classifier (81.6%), Random Forest (81.3%), Logistic Regression (80.9%) |

## App 1: Churn Predictor Implementation
- Uses trained Voting Classifier model
- Processes customer CSV data
- Calculates churn probability for each customer
- Displays risk levels and key factors visually

## App 2: Email Generator Implementation

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

## Future Improvements: ML + LLM Integration

1. **Predict Risk**: Voting Classifier identifies customers likely to churn
2. **Explain Why**: SHAP values determine the key factors for each customer
3. **Personalize Message**: LLM generates tailored emails addressing specific factors
4. **Interface**: Both manual toggles and automated CSV batch processing
5. **RAG**: Ground offers in actual Vodafone PDFs about available prices/offers etc.

This combines predictive power (ML), explainability (SHAP), and personalization (LLM) to create targeted retention campaigns that address each customer's unique reasons for potential churn.

## Sources:

- [Customer Churn Kaggle Notebook](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction/notebook#-8.-Machine-Learning-Model-Evaluations-and-Predictions)
- [DaisyUI](https://daisyui.com/)
- [FastHTML](https://docs.fastht.ml/)
  
