import modal
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from io import StringIO
from fasthtml.common import *
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

# Define app
app = modal.App("churn_predictor")

# Constants and directories
DATA_DIR = "/data"
MODELS_DIR = "/models"

# Create custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "python-fasthtml==0.12.0"
    )
)

# Look up model volume (this should be created by train_save_model.py)
try:
    model_volume = modal.Volume.lookup("churn_models", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Churn models volume not found. Please run train_save_model.py first.")

# Look up data volume for storing predictions
try:
    data_volume = modal.Volume.lookup("churn_data", create_if_missing=True)
except modal.exception.NotFoundError:
    data_volume = modal.Volume.persisted("churn_data")

# Function to load model and predict churn
@app.function(
    image=image,
    volumes={MODELS_DIR: model_volume, DATA_DIR: data_volume}
)
def predict_churn(csv_data: str) -> Dict[str, Any]:
    """
    Predict churn using the trained model.
    
    Args:
        csv_data: String containing CSV data
        
    Returns:
        Dictionary with predictions and customer data
    """
    print("🔍 Loading model...")
    model_path = os.path.join(MODELS_DIR, "voting_classifier_model.joblib")
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    
    # Load the model and feature names
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    
    print("📊 Processing CSV data...")
    # Load CSV data into DataFrame
    df = pd.read_csv(StringIO(csv_data))
    
    # Save customer IDs for output
    customer_ids = df['customerID'].values
    
    # Prepare the data for prediction (similar to how it was prepared for training)
    X = df.drop(['customerID'], axis=1, errors='ignore')
    
    # Make sure all expected features are present
    missing_cols = set(feature_names) - set(X.columns)
    if missing_cols:
        error_msg = f"Missing columns in input data: {missing_cols}"
        print(f"⚠️ {error_msg}")
        return {"error": error_msg}
    
    # Reorder columns to match training data
    X = X[feature_names]
    
    # Process categorical features
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    # Make predictions
    print("🔮 Making predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # Probability of churning (class 1)
    
    # Prepare results
    results = []
    for i, customer_id in enumerate(customer_ids):
        customer_data = df.iloc[i].to_dict()
        results.append({
            "customerID": customer_id,
            "will_churn": bool(predictions[i]),
            "churn_probability": float(probabilities[i]),
            "customer_data": customer_data
        })
    
    print(f"✅ Processed {len(results)} customers")
    
    return {
        "total_customers": len(results),
        "predicted_churn": int(sum(predictions)),
        "predicted_retain": int(len(predictions) - sum(predictions)),
        "results": results
    }

# Main FastHTML Server with defined routes
@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    cpu=1.0,
    timeout=3600
)
@modal.asgi_app()
def serve():
    """Main FastHTML Server for Churn Predictor Dashboard"""
    # Set up the FastHTML app with required headers
    fasthtml_app, rt = fast_app(
        hdrs=(
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@3.9.2/dist/full.css"),
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
            Script(src="https://unpkg.com/htmx.org@1.9.10"),
            # Add custom theme styles
            Style("""
                :root {
                --color-base-100: oklch(98% 0.002 247.839);
                --color-base-200: oklch(96% 0.003 264.542);
                --color-base-300: oklch(92% 0.006 264.531);
                --color-base-content: oklch(21% 0.034 264.665);
                --color-primary: oklch(0% 0 0);
                --color-primary-content: oklch(100% 0 0);
                --color-secondary: oklch(62% 0.214 259.815);
                --color-secondary-content: oklch(97% 0.014 254.604);
                --color-accent: oklch(62% 0.214 259.815);
                --color-accent-content: oklch(97% 0.014 254.604);
                --color-neutral: oklch(13% 0.028 261.692);
                --color-neutral-content: oklch(98% 0.002 247.839);
                --color-info: oklch(58% 0.158 241.966);
                --color-info-content: oklch(97% 0.013 236.62);
                --color-success: oklch(62% 0.194 149.214);
                --color-success-content: oklch(98% 0.018 155.826);
                --color-warning: oklch(66% 0.179 58.318);
                --color-warning-content: oklch(98% 0.022 95.277);
                --color-error: oklch(59% 0.249 0.584);
                --color-error-content: oklch(97% 0.014 343.198);
                }

                /* Custom styling for risk levels */
                .high-risk {
                    background-color: rgba(239, 68, 68, 0.1);
                }
                
                .medium-risk {
                    background-color: rgba(245, 158, 11, 0.1);
                }
                
                .low-risk {
                    background-color: rgba(16, 185, 129, 0.1);
                }
            """),
        )
    )
    
    #################################################
    # Homepage Route - Churn Predictor Dashboard
    #################################################
    @rt("/")
    def homepage():
        """Render the churn predictor dashboard"""
        
        # File upload section
        file_upload = Div(
            H2("Upload Customer Data", cls="text-xl font-bold mb-4 text-purple-700"),
            P("Upload a CSV file with customer data to predict churn risk.", cls="mb-4"),
            Form(
                Div(
                    Label("Select CSV File", cls="block text-sm font-medium mb-2"),
                    Input(
                        type="file",
                        name="csv_file",
                        accept=".csv",
                        cls="w-full p-2 border border-gray-300 rounded",
                        id="csv-file-input"
                    ),
                    cls="mb-4"
                ),
                Div(
                    Button(
                        Span(cls="loading loading-spinner loading-xs mr-2 hidden", id="button-spinner"),
                        "Predict Churn",
                        cls="btn btn-primary btn-md w-auto px-6",
                        id="predict-button",
                        disabled="disabled",
                        type="button"
                    ),
                    cls="flex items-center"
                ),
                Script("""
                document.addEventListener('DOMContentLoaded', function() {
                    console.log('DOM loaded');
                    const fileInput = document.getElementById('csv-file-input');
                    const predictButton = document.getElementById('predict-button');
                    const buttonSpinner = document.getElementById('button-spinner');
                    const resultsContainer = document.getElementById('results-container');
                    
                    // Enable predict button when file is selected
                    fileInput.addEventListener('change', function() {
                        console.log('File selected');
                        predictButton.disabled = !fileInput.files.length;
                    });
                    
                    // Handle button click
                    predictButton.addEventListener('click', function() {
                        console.log('Button clicked');
                        
                        // Show loading spinner in button
                        buttonSpinner.classList.remove('hidden');
                        predictButton.classList.add('loading-btn');
                        predictButton.disabled = true;
                        
                        // Create form data
                        const formData = new FormData();
                        formData.append('csv_file', fileInput.files[0]);
                        
                        // Send request
                        console.log('Sending fetch request');
                        fetch('/predict', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => {
                            console.log('Response received');
                            return response.text();
                        })
                        .then(html => {
                            console.log('Processing HTML');
                            
                            // Hide loading spinner
                            buttonSpinner.classList.add('hidden');
                            predictButton.classList.remove('loading-btn');
                            predictButton.disabled = false;
                            
                            // Update results container with response HTML
                            resultsContainer.innerHTML = html;
                        })
                        .catch(error => {
                            console.error('Error predicting churn:', error);
                            
                            // Hide loading spinner
                            buttonSpinner.classList.add('hidden');
                            predictButton.classList.remove('loading-btn');
                            predictButton.disabled = false;
                            
                            alert('Error processing request. Please try again.');
                        });
                    });
                });
                """),
                id="upload-form",
                cls="bg-base-200 p-6 rounded-lg shadow-lg border mb-6"
            ),
            cls="mb-8"
        )
        
        # Results container (will be populated via HTMX)
        results_container = Div(
            # Initially empty, will be filled by HTMX
            P("Upload a CSV file to predict customer churn.", cls="text-center text-gray-500"),
            id="results-container",
            cls="bg-base-200 p-6 rounded-lg shadow-lg border"
        )
        
        return Title("Telecom Churn Predictor"), Main(
            Div(
                H1("Telecom Churn Predictor", cls="text-2xl font-bold text-center mb-6 text-purple-700"),
                P("Upload customer data to predict which customers are at risk of churning.", 
                  cls="text-center mb-8"),
                file_upload,
                results_container,
                cls="container mx-auto px-4 py-8 max-w-6xl"
            ),
            cls="min-h-screen bg-base-100",
            data_theme="light"
        )
    
    #################################################
    # Predict Route - Process CSV and Return Results
    #################################################
    @rt("/predict", methods=["POST"])
    async def predict_endpoint(request):
        """Process CSV and return results HTML"""
        try:
            # Get CSV file from form data
            form_data = await request.form()
            csv_file = form_data.get('csv_file')
            
            if not csv_file:
                return HTMLResponse("<div class='alert alert-error'>No CSV file provided</div>")
            
            # Read CSV content
            csv_content = await csv_file.read()
            csv_text = csv_content.decode('utf-8')
            
            # Call prediction function
            prediction_data = predict_churn.remote(csv_text)
            
            if "error" in prediction_data:
                return HTMLResponse(f"<div class='alert alert-error'>{prediction_data['error']}</div>")
            
            # Create summary cards HTML
            summary_html = f"""
            <h2 class="text-xl font-bold mb-4 text-purple-700">Churn Prediction Summary</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div class="bg-base-200 p-4 rounded">
                    <h3 class="text-lg font-semibold">Total Customers</h3>
                    <p class="text-2xl font-bold">{prediction_data['total_customers']}</p>
                </div>
                <div class="bg-base-200 p-4 rounded">
                    <h3 class="text-lg font-semibold text-error">Predicted to Churn</h3>
                    <p class="text-2xl font-bold">{prediction_data['predicted_churn']}</p>
                </div>
                <div class="bg-base-200 p-4 rounded">
                    <h3 class="text-lg font-semibold text-success">Predicted to Stay</h3>
                    <p class="text-2xl font-bold">{prediction_data['predicted_retain']}</p>
                </div>
            </div>
            """
            
            # Create table header HTML
            table_html = """
            <h2 class="text-xl font-bold mb-4 text-purple-700">Customer Prediction Results</h2>
            <div class="overflow-x-auto">
                <table class="table table-zebra w-full">
                    <thead>
                        <tr>
                            <th>Customer ID</th>
                            <th>Prediction</th>
                            <th>Churn Risk</th>
                            <th>Tenure</th>
                            <th>Monthly Charges</th>
                            <th>Contract</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # Add table rows
            for customer in prediction_data["results"]:
                # Determine risk class and icon
                risk_class = ""
                if customer["churn_probability"] > 0.7:
                    risk_class = "high-risk"
                elif customer["churn_probability"] > 0.4:
                    risk_class = "medium-risk"
                else:
                    risk_class = "low-risk"
                
                churn_icon = "🚩" if customer["will_churn"] else "✅"
                
                risk_text_class = ""
                if customer["churn_probability"] > 0.7:
                    risk_text_class = "text-error"
                elif customer["churn_probability"] > 0.4:
                    risk_text_class = "text-warning"
                else:
                    risk_text_class = "text-success"
                
                # Format risk percentage
                risk_percent = int(customer["churn_probability"] * 100)
                
                # Add row to table
                table_html += f"""
                <tr class="{risk_class}">
                    <td>{customer['customerID']}</td>
                    <td><span class="text-xl">{churn_icon}</span></td>
                    <td><span class="{risk_text_class} font-bold">{risk_percent}%</span></td>
                    <td>{customer['customer_data']['tenure']}</td>
                    <td>${customer['customer_data']['MonthlyCharges']}</td>
                    <td>{customer['customer_data']['Contract']}</td>
                </tr>
                """
            
            # Close table
            table_html += """
                    </tbody>
                </table>
            </div>
            """
            
            # Combine HTML and return
            return HTMLResponse(summary_html + table_html)
            
        except Exception as e:
            print(f"Error predicting churn: {e}")
            return HTMLResponse(f"<div class='alert alert-error'>Error processing request: {str(e)}</div>")
    
    # Return the FastHTML app
    return fasthtml_app

if __name__ == "__main__":
    print("Starting Telecom Churn Predictor...")
