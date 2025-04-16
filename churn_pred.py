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
                        "Predict Churn",
                        cls="btn btn-primary",
                        id="predict-button",
                        disabled="disabled"
                    ),
                    Div(
                        cls="loading loading-spinner loading-md text-primary ml-4 hidden",
                        id="loading-indicator"
                    ),
                    cls="flex items-center"
                ),
                id="upload-form",
                method="POST",
                enctype="multipart/form-data",
                cls="bg-base-200 p-6 rounded-lg shadow-lg border mb-6"
            ),
            cls="mb-8"
        )
        
        # Summary section (initially hidden)
        summary_section = Div(
            H2("Churn Prediction Summary", cls="text-xl font-bold mb-4 text-purple-700"),
            Div(
                Div(
                    H3("Total Customers", cls="text-lg font-semibold"),
                    P("0", id="total-customers", cls="text-2xl font-bold"),
                    cls="bg-base-200 p-4 rounded"
                ),
                Div(
                    H3("Predicted to Churn", cls="text-lg font-semibold text-error"),
                    P("0", id="predicted-churn", cls="text-2xl font-bold"),
                    cls="bg-base-200 p-4 rounded"
                ),
                Div(
                    H3("Predicted to Stay", cls="text-lg font-semibold text-success"),
                    P("0", id="predicted-retain", cls="text-2xl font-bold"),
                    cls="bg-base-200 p-4 rounded"
                ),
                cls="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6"
            ),
            cls="bg-base-200 p-6 rounded-lg shadow-lg border mb-6 hidden",
            id="summary-section"
        )
        
        # Results section (initially hidden)
        results_section = Div(
            H2("Customer Prediction Results", cls="text-xl font-bold mb-4 text-purple-700"),
            Table(
                Thead(
                    Tr(
                        Th("Customer ID"),
                        Th("Prediction"),
                        Th("Churn Risk"),
                        Th("Tenure"),
                        Th("Monthly Charges"),
                        Th("Contract"),
                        Th("Actions")
                    )
                ),
                Tbody(
                    id="results-table-body"
                ),
                cls="table table-zebra w-full"
            ),
            cls="bg-base-200 p-6 rounded-lg shadow-lg border hidden",
            id="results-section"
        )
        
        # Customer details modal
        customer_modal = Div(
            Div(
                Div(
                    H3("Customer Details", cls="text-lg font-bold", id="modal-customer-id"),
                    Button(
                        "✕",
                        cls="btn btn-sm btn-circle absolute right-2 top-2",
                        onclick="document.getElementById('customer-modal').classList.add('hidden')"
                    ),
                    cls="modal-header p-4 border-b"
                ),
                Div(
                    id="modal-content",
                    cls="p-4"
                ),
                Div(
                    Button(
                        "Close",
                        cls="btn",
                        onclick="document.getElementById('customer-modal').classList.add('hidden')"
                    ),
                    cls="modal-footer p-4 border-t"
                ),
                cls="modal-box relative"
            ),
            cls="modal hidden",
            id="customer-modal"
        )
        
        # Add script for form handling
        form_script = Script("""
        document.addEventListener('DOMContentLoaded', function() {
            const fileInput = document.getElementById('csv-file-input');
            const predictButton = document.getElementById('predict-button');
            const loadingIndicator = document.getElementById('loading-indicator');
            const summarySection = document.getElementById('summary-section');
            const resultsSection = document.getElementById('results-section');
            
            // Enable predict button when file is selected
            fileInput.addEventListener('change', function() {
                predictButton.disabled = !fileInput.files.length;
            });
            
            // Handle form submission
            document.getElementById('upload-form').addEventListener('submit', function(event) {
                event.preventDefault();
                
                // Show loading indicator
                loadingIndicator.classList.remove('hidden');
                predictButton.disabled = true;
                
                // Create form data
                const formData = new FormData();
                formData.append('csv_file', fileInput.files[0]);
                
                // Send request
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    loadingIndicator.classList.add('hidden');
                    predictButton.disabled = false;
                    
                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }
                    
                    // Update summary
                    document.getElementById('total-customers').innerText = data.total_customers;
                    document.getElementById('predicted-churn').innerText = data.predicted_churn;
                    document.getElementById('predicted-retain').innerText = data.predicted_retain;
                    summarySection.classList.remove('hidden');
                    
                    // Update results table
                    const tableBody = document.getElementById('results-table-body');
                    tableBody.innerHTML = '';
                    
                    data.results.forEach(customer => {
                        const row = document.createElement('tr');
                        
                        // Set row class based on churn probability
                        if (customer.churn_probability > 0.7) {
                            row.className = 'high-risk';
                        } else if (customer.churn_probability > 0.4) {
                            row.className = 'medium-risk';
                        } else {
                            row.className = 'low-risk';
                        }
                        
                        // Create customer ID cell
                        const idCell = document.createElement('td');
                        idCell.innerText = customer.customerID;
                        row.appendChild(idCell);
                        
                        // Create prediction cell
                        const predictionCell = document.createElement('td');
                        predictionCell.innerHTML = customer.will_churn ? 
                            '<span class="text-xl">🚩</span>' : 
                            '<span class="text-xl">✅</span>';
                        row.appendChild(predictionCell);
                        
                        // Create risk cell
                        const riskCell = document.createElement('td');
                        const riskPercent = Math.round(customer.churn_probability * 100);
                        const riskColor = riskPercent > 70 ? 'text-error' : 
                                         riskPercent > 40 ? 'text-warning' : 'text-success';
                        riskCell.innerHTML = `<span class="${riskColor} font-bold">${riskPercent}%</span>`;
                        row.appendChild(riskCell);
                        
                        // Create tenure cell
                        const tenureCell = document.createElement('td');
                        tenureCell.innerText = customer.customer_data.tenure;
                        row.appendChild(tenureCell);
                        
                        // Create monthly charges cell
                        const chargesCell = document.createElement('td');
                        chargesCell.innerText = '$' + customer.customer_data.MonthlyCharges;
                        row.appendChild(chargesCell);
                        
                        // Create contract cell
                        const contractCell = document.createElement('td');
                        contractCell.innerText = customer.customer_data.Contract;
                        row.appendChild(contractCell);
                        
                        // Create actions cell
                        const actionsCell = document.createElement('td');
                        const viewButton = document.createElement('button');
                        viewButton.className = 'btn btn-xs btn-outline';
                        viewButton.innerText = 'View';
                        viewButton.addEventListener('click', function() {
                            showCustomerDetails(customer);
                        });
                        actionsCell.appendChild(viewButton);
                        row.appendChild(actionsCell);
                        
                        tableBody.appendChild(row);
                    });
                    
                    resultsSection.classList.remove('hidden');
                })
                .catch(error => {
                    console.error('Error predicting churn:', error);
                    loadingIndicator.classList.add('hidden');
                    predictButton.disabled = false;
                    alert('Error processing request. Please try again.');
                });
            });
            
            // Function to show customer details in modal
            window.showCustomerDetails = function(customer) {
                const modal = document.getElementById('customer-modal');
                const modalCustomerId = document.getElementById('modal-customer-id');
                const modalContent = document.getElementById('modal-content');
                
                modalCustomerId.innerText = `Customer: ${customer.customerID}`;
                
                // Build content HTML
                let contentHTML = `
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div class="p-4 border rounded">
                            <h4 class="font-bold mb-2">Churn Prediction</h4>
                            <div class="text-2xl font-bold ${customer.will_churn ? 'text-error' : 'text-success'}">
                                ${customer.will_churn ? '🚩 Likely to Churn' : '✅ Likely to Stay'}
                            </div>
                            <div class="mt-2">
                                <span class="font-bold">Risk Level:</span> ${Math.round(customer.churn_probability * 100)}%
                            </div>
                        </div>
                        <div class="p-4 border rounded">
                            <h4 class="font-bold mb-2">Customer Overview</h4>
                            <div><span class="font-bold">Gender:</span> ${customer.customer_data.gender}</div>
                            <div><span class="font-bold">Senior Citizen:</span> ${customer.customer_data.SeniorCitizen ? 'Yes' : 'No'}</div>
                            <div><span class="font-bold">Partner:</span> ${customer.customer_data.Partner}</div>
                            <div><span class="font-bold">Dependents:</span> ${customer.customer_data.Dependents}</div>
                        </div>
                    </div>
                    
                    <div class="mb-4">
                        <h4 class="font-bold mb-2">Service Details</h4>
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div class="p-4 border rounded">
                                <div><span class="font-bold">Tenure:</span> ${customer.customer_data.tenure} months</div>
                                <div><span class="font-bold">Contract:</span> ${customer.customer_data.Contract}</div>
                                <div><span class="font-bold">Payment Method:</span> ${customer.customer_data.PaymentMethod}</div>
                                <div><span class="font-bold">Paperless Billing:</span> ${customer.customer_data.PaperlessBilling}</div>
                            </div>
                            <div class="p-4 border rounded">
                                <div><span class="font-bold">Phone Service:</span> ${customer.customer_data.PhoneService}</div>
                                <div><span class="font-bold">Multiple Lines:</span> ${customer.customer_data.MultipleLines}</div>
                                <div><span class="font-bold">Internet Service:</span> ${customer.customer_data.InternetService}</div>
                            </div>
                            <div class="p-4 border rounded">
                                <div><span class="font-bold">Online Security:</span> ${customer.customer_data.OnlineSecurity}</div>
                                <div><span class="font-bold">Online Backup:</span> ${customer.customer_data.OnlineBackup}</div>
                                <div><span class="font-bold">Device Protection:</span> ${customer.customer_data.DeviceProtection}</div>
                                <div><span class="font-bold">Tech Support:</span> ${customer.customer_data.TechSupport}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="mb-4">
                        <h4 class="font-bold mb-2">Financial Information</h4>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div class="p-4 border rounded">
                                <div class="text-xl font-bold">Monthly Charges: $${customer.customer_data.MonthlyCharges}</div>
                                <div>Total Charges: $${customer.customer_data.TotalCharges}</div>
                            </div>
                        </div>
                    </div>
                `;
                
                modalContent.innerHTML = contentHTML;
                modal.classList.remove('hidden');
            }
        });
        """)
        
        return Title("Telecom Churn Predictor"), Main(
            form_script,
            Div(
                H1("Telecom Churn Predictor", cls="text-2xl font-bold text-center mb-6 text-purple-700"),
                P("Upload customer data to predict which customers are at risk of churning.", 
                  cls="text-center mb-8"),
                file_upload,
                summary_section,
                results_section,
                customer_modal,
                cls="container mx-auto px-4 py-8 max-w-6xl"
            ),
            cls="min-h-screen bg-base-100",
            data_theme="light"
        )
    
    #################################################
    # Predict API Endpoint
    #################################################
    @rt("/predict", methods=["POST"])
    async def api_predict_churn(request):
        """API endpoint to process CSV and predict churn"""
        try:
            # Get CSV file from form data
            form_data = await request.form()
            csv_file = form_data.get('csv_file')
            
            if not csv_file:
                return JSONResponse({"error": "No CSV file provided"}, status_code=400)
            
            # Read CSV content
            csv_content = await csv_file.read()
            csv_text = csv_content.decode('utf-8')
            
            # Call prediction function
            result = predict_churn.remote(csv_text)
            
            return JSONResponse(result)
            
        except Exception as e:
            print(f"Error predicting churn: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    # Return the FastHTML app
    return fasthtml_app

if __name__ == "__main__":
    print("Starting Telecom Churn Predictor...")