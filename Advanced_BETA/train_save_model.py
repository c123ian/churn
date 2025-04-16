import modal
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Define app!
app = modal.App("churn_model_uploader")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "shap"
    )
)

# Look up or create volume
model_volume = modal.Volume.lookup("churn_models", create_if_missing=True)



@app.function(image=image,cpu=1.0,volumes={"/models": model_volume})
def train_and_save_model():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("Vodafone_Customer_Churn_Sample_Dataset.csv")
    
    # Prepare features and target
    print("Preparing data...")
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Preprocess data (simplified for example)
    # Convert categorical columns
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train voting classifier
    print("Training voting classifier...")
    clf1 = GradientBoostingClassifier()
    clf2 = LogisticRegression()
    clf3 = AdaBoostClassifier()
    
    voting_model = VotingClassifier(
        estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], 
        voting='soft'
    )
    
    voting_model.fit(X_train, y_train)
    
    # Save model to volume
    print("Saving model to volume...")
    joblib.dump(voting_model, '/models/voting_classifier_model.joblib')
    
    # Also save feature names for SHAP analysis later
    joblib.dump(list(X.columns), '/models/feature_names.joblib')
    
    print("Model saved successfully!")
    return "Complete"

if __name__ == "__main__":
    with modal.Runner() as runner:
        runner.run(train_and_save_model)
