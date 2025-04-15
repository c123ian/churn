import modal
import os
import sqlite3
import uuid
import time
import random
import json
from typing import Optional, Dict, Any, List
import torch
import base64

from fasthtml.common import *
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

# Define app
app = modal.App("email_generator")

# Constants and directories
DATA_DIR = "/data"
EMAILS_FOLDER = "/data/generated_emails"
DB_PATH = "/data/email_generator.db"
STATUS_DIR = "/data/status"
LLAMA_DIR = "/llama_mini"  # Path to where Llama model is stored

# Create custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "transformers==4.46.1",
        "accelerate>=0.26.0",
        "torch>=2.0.0",
        "sentencepiece",
        "python-fasthtml==0.12.0"
    )
)

# Look up Llama model volume
try:
    llm_volume = modal.Volume.lookup("llama_mini", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Llama model volume not found. Please ensure it's set up correctly.")

# Look up data volume for storing emails
try:
    email_volume = modal.Volume.lookup("email_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    email_volume = modal.Volume.persisted("email_volume")

# Generation prompt template - Updated to add Final_Answer instruction and exact signature format
EMAIL_GENERATION_PROMPT = """
You are an expert in customer retention for a telecom company. Your task is to generate a personalized email 
for a customer at risk of churning based on their profile information.

Follow Vodafone's tone of voice guidelines:
- Friendly and approachable
- Clear and concise
- Positive and reassuring
- Professional and trustworthy

Use this structure:
1. Subject line (eye-catching, personalized)
2. Personalized greeting
3. Brief introduction (expressing appreciation for their business)
4. Body with bullet points for key benefits (based on the customer profile)
5. Clear call to action
6. Warm closing
7. Professional signature

The email should be tailored based on the customer profile that follows:
{customer_profile}

Include specific retention strategies from this list where appropriate:
{retention_strategies}

IMPORTANT FORMATTING RULES:
1. Always end with the final draft of the email as Final_Answer: followed by the complete email.
2. Always end your email with EXACTLY the following signature format:
   Warm regards,

   Sarah Thompson
   Customer Retention Team
   Vodafone

3. Do not include ANY text after the signature.
"""

# Mapping of risk factors to retention strategies
RISK_FACTOR_STRATEGIES = {
    "highMonthlyCharges": [
        "Discount on 1-year contract", 
        "Loyalty price reduction",
        "Bundle discount"
    ],
    "fiberOptic": [
        "Free speed upgrade",
        "Enhanced reliability promise",
        "Free premium router"
    ],
    "noDependents": [
        "Single-user special rates",
        "Personalized usage plan"
    ],
    "noPartner": [
        "Friend referral bonus",
        "Solo user benefits"
    ],
    "seniorCitizen": [
        "Senior discount",
        "Simplified service package",
        "Dedicated support line"
    ],
    "noOnlineSecurity": [
        "Free security package for 6 months",
        "Discounted security bundle"
    ],
    "paperlessBilling": [
        "Loyalty discount for paperless users"
    ],
    "noTechSupport": [
        "Free tech support trial",
        "One-time free tech visit"
    ],
    "monthToMonth": [
        "Significant discount for 1-year commitment",
        "Flexible exit terms"
    ],
    "electronicCheck": [
        "Auto-pay discount",
        "Payment method bonus"
    ],
    "lowTenure": [
        "Early loyalty rewards",
        "New customer success program"
    ],
    "noMultipleLines": [
        "Additional line special offer",
        "Family plan introduction"
    ],
    "noStreamingServices": [
        "Free streaming service trial",
        "Bundled entertainment package"
    ]
}

# Function to extract email with specific start and end boundaries
def extract_final_answer(text):
    """
    Extract the content between specific boundaries:
    - Start: After "Final_Answer:" marker
    - End: At the end of "Customer Retention Team\nVodafone" signature
    
    This ensures we only display the final formatted email without any additional text.
    """
    # First, extract content after Final_Answer marker
    if "Final_Answer:" in text:
        email_text = text.split("Final_Answer:", 1)[1].strip()
    else:
        email_text = text  # Use original if marker not found
    
    # Next, ensure we cut off at the end of the signature
    if "Customer Retention Team" in email_text:
        # Find the signature and include "Vodafone" line that follows it
        signature_pos = email_text.find("Customer Retention Team")
        end_pos = email_text.find("\n", signature_pos)
        
        # If there's content after "Vodafone", trim it off
        if end_pos > 0:
            # Include the line with "Vodafone" but nothing after
            vodafone_pos = email_text.find("Vodafone", signature_pos)
            if vodafone_pos > 0:
                end_pos = email_text.find("\n", vodafone_pos)
                if end_pos > 0:
                    email_text = email_text[:end_pos]
                else:
                    # If no newline after Vodafone, include to the end
                    pass
        
    # Return the properly bounded email
    return email_text

# Function to save email to file
def save_email_file(email_id, customer_profile, email_content):
    """Save generated email to a file"""
    os.makedirs(EMAILS_FOLDER, exist_ok=True)
    email_file = os.path.join(EMAILS_FOLDER, f"{email_id}.json")
    email_data = {
        "id": email_id,
        "profile": customer_profile,
        "email": email_content,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open(email_file, "w") as f:
            json.dump(email_data, f)
        print(f"✅ Saved email file for ID: {email_id}")
        return True
    except Exception as e:
        print(f"⚠️ Error saving email file: {e}")
        return False

# Setup database for email generation
def setup_database(db_path: str):
    """Initialize SQLite database for emails"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrency
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL,
            customer_profile TEXT NOT NULL,
            email_content TEXT,
            status TEXT DEFAULT 'generated',
            feedback TEXT DEFAULT NULL, 
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

# Generate email using Llama model
@app.function(
    image=image,
    gpu=modal.gpu.A10G(count=1),
    timeout=300,
    volumes={LLAMA_DIR: llm_volume, DATA_DIR: email_volume}
)
def generate_email_llm(customer_profile: Dict[str, Any]) -> str:
    """Generate email using Llama model based on customer profile"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from accelerate import Accelerator
    
    print(f"🚀 Generating email with LLM for customer: {customer_profile.get('customerName', 'Unknown')}")
    
    email_id = uuid.uuid4().hex
    
    # Extract customer name
    customer_name = customer_profile.get("customerName", "Valued Customer")
    
    # Extract risk factors and map to strategies
    risk_factors = []
    retention_strategies = []
    
    # Handle monthly charges
    monthly_charges = customer_profile.get("monthlyCharges", 0)
    if monthly_charges > 75:
        risk_factors.append(f"High monthly charges (${monthly_charges})")
        retention_strategies.extend(RISK_FACTOR_STRATEGIES["highMonthlyCharges"])
    elif monthly_charges > 50:
        risk_factors.append(f"Medium monthly charges (${monthly_charges})")
        retention_strategies.append("Bundle discount")
    
    # Handle other boolean factors
    factor_mapping = {
        "fiberOptic": "Has Fiber Optic service",
        "noDependents": "No dependents (single user)",
        "noPartner": "No partner",
        "seniorCitizen": "Senior citizen",
        "noOnlineSecurity": "No online security",
        "paperlessBilling": "Uses paperless billing",
        "noTechSupport": "No tech support",
        "monthToMonth": "Month-to-month contract",
        "electronicCheck": "Uses electronic check payment",
        "lowTenure": "New customer (low tenure)",
        "noMultipleLines": "No multiple lines",
        "noStreamingServices": "No streaming services"
    }
    
    for factor_key, description in factor_mapping.items():
        if customer_profile.get(factor_key, False):
            risk_factors.append(description)
            if factor_key in RISK_FACTOR_STRATEGIES:
                retention_strategies.extend(RISK_FACTOR_STRATEGIES[factor_key])
    
    # Format the profile and strategies for the prompt
    formatted_profile = "\n".join([f"- {factor}" for factor in risk_factors])
    formatted_strategies = "\n".join(retention_strategies)
    
    # Prepare the prompt with customer name
    prompt = EMAIL_GENERATION_PROMPT.format(
        customer_profile=f"Customer Name: {customer_name}\n{formatted_profile}",
        retention_strategies=formatted_strategies
    )
    
    # Load Llama model
    print("Loading Llama model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_DIR,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    # Create generation pipeline
    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=800,
        temperature=0.7
    )
    
    # Generate the email
    print("Generating email content...")
    output = generation_pipe(prompt)
    raw_email_content = output[0]["generated_text"][len(prompt):].strip()
    
    # Extract just the email content after Final_Answer:
    email_content = extract_final_answer(raw_email_content)
    
    # Save the generated email
    profile_json = json.dumps(customer_profile)
    save_email_file(email_id, profile_json, email_content)
    
    # Store in database
    try:
        conn = setup_database(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO emails (id, customer_name, customer_profile, email_content) VALUES (?, ?, ?, ?)",
            (email_id, customer_name, profile_json, email_content)
        )
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Error saving to database: {e}")
        raise e  # Re-raise to ensure errors bubble up
    
    return email_content

# Main FastHTML Server with defined routes
@app.function(
    image=image,
    volumes={DATA_DIR: email_volume},
    cpu=1.0,
    timeout=3600
)
@modal.asgi_app()
def serve():
    """Main FastHTML Server for Email Generator Dashboard"""
    # Set up the FastHTML app with required headers
    fasthtml_app, rt = fast_app(
        hdrs=(
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@3.9.2/dist/full.css"),
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
            Script(src="https://unpkg.com/htmx.org@1.9.10"),
            # Add custom theme styles from https://daisyui.com/theme-generator/
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
                --radius-selector: 0rem;
                --radius-field: 0.5rem;
                --radius-box: 2rem;
                --size-selector: 0.25rem;
                --size-field: 0.25rem;
                --border: 1px;
                }

                /* Custom styling for better contrast */
                .text-purple-700 {
                    color: oklch(54% 0.25 280);
                }
                
                .bg-custom-dark {
                    background-color: oklch(37% 0 0);
                }
                
                .custom-border {
                    border-color: var(--color-base-300);
                }
            """),
        )
    )
    
    # Ensure database exists
    setup_database(DB_PATH)
    
    #################################################
    # Homepage Route - Email Generator Dashboard
    #################################################
    @rt("/")
    def homepage():
        """Render the email generator dashboard"""
        
        # Customer name input
        customer_name_input = Div(
            Label("Customer Name", cls="block text-sm font-medium mb-2"),
            Input(
                type="text",
                name="customerName",
                value="Alex Johnson",
                cls="w-full p-2 border border-gray-300 rounded"
            ),
            cls="mb-4"
        )
        
        # Monthly charges slider
        monthly_charges_slider = Div(
            Label("Monthly Charges: $<span id='charges-value'>50</span>", cls="block text-sm font-medium mb-2"),
            Div(
                Input(
                    type="range",
                    name="monthlyCharges",
                    min="0",
                    max="100",
                    value="50",
                    cls="range",
                    step="25",
                    id="charges-slider",
                    hx_on_input="document.getElementById('charges-value').innerText = this.value"
                ),
                Div(
                    Span("|", cls="text-xs"),
                    Span("|", cls="text-xs"),
                    Span("|", cls="text-xs"),
                    Span("|", cls="text-xs"),
                    Span("|", cls="text-xs"),
                    cls="flex justify-between px-2.5 mt-2 text-xs"
                ),
                Div(
                    Span("Very Low", cls="text-xs"),
                    Span("Low", cls="text-xs"),
                    Span("Medium", cls="text-xs"),
                    Span("High", cls="text-xs"),
                    Span("Very High", cls="text-xs"),
                    cls="flex justify-between px-2.5 mt-2 text-xs"
                ),
                cls="w-full max-w-xs"
            ),
            cls="mb-6"
        )
        
        # Create toggle switches for risk factors
        def create_toggle(name, label, category="primary", checked=False):
            return Div(
                Label(label, cls="text-sm"),
                Input(
                    type="checkbox",
                    name=name,
                    checked="checked" if checked else None,
                    cls=f"toggle toggle-{category}"
                ),
                cls="mb-2 flex items-center justify-between"
            )
        
        # Service features section
        service_features = Div(
            H3("Service Features", cls="text-lg font-semibold mb-2"),
            create_toggle("fiberOptic", "Fiber Optic"),
            create_toggle("monthToMonth", "Month-to-Month Contract", checked=True),
            create_toggle("paperlessBilling", "Paperless Billing", checked=True),
            create_toggle("electronicCheck", "Electronic Check Payment"),
            cls="mb-6"
        )
        
        # Customer demographics section
        customer_demographics = Div(
            H3("Customer Demographics", cls="text-lg font-semibold mb-2"),
            create_toggle("noDependents", "No Dependents (Single User)", "secondary"),
            create_toggle("noPartner", "No Partner", "secondary"),
            create_toggle("seniorCitizen", "Senior Citizen", "secondary"),
            create_toggle("lowTenure", "New Customer (Low Tenure)", "secondary"),
            cls="mb-6"
        )
        
        # Missing services section
        missing_services = Div(
            H3("Missing Services", cls="text-lg font-semibold mb-2"),
            Div(
                create_toggle("noOnlineSecurity", "No Online Security", "error"),
                create_toggle("noTechSupport", "No Tech Support", "error"),
                create_toggle("noMultipleLines", "No Multiple Lines", "error", checked=True),
                create_toggle("noStreamingServices", "No Streaming Services", "error"),
                cls="grid grid-cols-1 md:grid-cols-2 gap-4"
            ),
            cls="mb-6"
        )
        
        # Controls panel
        controls_panel = Div(
            H2("Customer Profile", cls="text-xl font-bold mb-4 text-purple-700"),
            customer_name_input,
            monthly_charges_slider,
            Div(
                service_features,
                customer_demographics,
                cls="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6"
            ),
            missing_services,
            Button(
                "Generate Email",
                cls="btn btn-primary w-full",
                id="generate-button",
                hx_post="/generate-email",
                hx_target="#email-content",
                hx_indicator="#loading-indicator"
            ),
            cls="w-full md:w-1/2 bg-base-100 p-6 rounded-lg shadow-lg custom-border border"
        )
        
        # Email preview panel with feedback buttons
        email_preview = Div(
            H2("Generated Email", cls="text-xl font-bold mb-4 text-purple-700"),
            Div(
                Div(
                    cls="loading loading-spinner loading-lg text-primary",
                    id="loading-indicator"
                ),
                cls="flex justify-center items-center h-12 hidden"
            ),
            Input(
                type="hidden",
                id="current-email-id",
                name="current_email_id",
                value=""
            ),
            Textarea(
                placeholder="Your generated email will appear here...",
                cls="textarea textarea-bordered w-full h-96 bg-base-200 text-base-content",
                id="email-content",
                readonly="readonly"
            ),
            Div(
                Div(
                    Span("Rate this email:", cls="text-sm mr-2"),
                    Button(
                        "👍",
                        cls="btn btn-outline btn-sm mr-2",
                        id="thumbs-up-button"
                    ),
                    Button(
                        "👎",
                        cls="btn btn-outline btn-sm mr-4",
                        id="thumbs-down-button"
                    ),
                    Span("", id="feedback-message", cls="text-sm"),
                    cls="flex items-center"
                ),
                Div(
                    Button(
                        "Copy to Clipboard",
                        cls="btn btn-outline btn-accent btn-sm",
                        hx_on_click="navigator.clipboard.writeText(document.getElementById('email-content').value); this.innerText = 'Copied!'; setTimeout(() => this.innerText = 'Copy to Clipboard', 2000)"
                    ),
                    cls="ml-auto"
                ),
                cls="mt-4 flex justify-between items-center"
            ),
            cls="w-full md:w-1/2 bg-base-100 p-6 rounded-lg shadow-lg custom-border border"
        )
        
        # Add script for form handling
        form_script = Script("""
        document.addEventListener('DOMContentLoaded', function() {
            // Function to gather form data
            function getFormData() {
                const formData = {
                    customerName: document.querySelector('input[name="customerName"]').value,
                    monthlyCharges: parseInt(document.querySelector('input[name="monthlyCharges"]').value)
                };
                
                // Get all toggle values
                const toggles = document.querySelectorAll('input[type="checkbox"]');
                toggles.forEach(toggle => {
                    formData[toggle.name] = toggle.checked;
                });
                
                return formData;
            }
            
            // Generate email when the button is clicked
            document.getElementById('generate-button').addEventListener('click', function(event) {
                event.preventDefault(); // Prevent default form submission
                
                const formData = getFormData();
                
                // Show loading indicator
                document.getElementById('loading-indicator').parentElement.classList.remove('hidden');
                
                fetch('/generate-email', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                })
                .then(response => response.json())  // Now expecting JSON response
                .then(data => {
                    // Update email content
                    document.getElementById('email-content').value = data.email_content;
                    
                    // Store email ID in hidden field
                    document.getElementById('current-email-id').value = data.email_id;
                    
                    // Reset feedback state
                    document.getElementById('thumbs-up-button').disabled = false;
                    document.getElementById('thumbs-down-button').disabled = false;
                    document.getElementById('feedback-message').innerText = '';
                    
                    // Hide loading indicator
                    document.getElementById('loading-indicator').parentElement.classList.add('hidden');
                })
                .catch(error => {
                    console.error('Error generating email:', error);
                    document.getElementById('email-content').value = 'Error generating email. Please try again.';
                    document.getElementById('loading-indicator').parentElement.classList.add('hidden');
                });
            });
            
            // Update the charges value display without triggering email generation
            document.getElementById('charges-slider').addEventListener('input', function() {
                document.getElementById('charges-value').innerText = this.value;
            });
            
            // Handle feedback button clicks
            document.getElementById('thumbs-up-button').addEventListener('click', function() {
                submitFeedback('positive');
            });
            
            document.getElementById('thumbs-down-button').addEventListener('click', function() {
                submitFeedback('negative');
            });
            
            // Function to submit feedback
            function submitFeedback(feedbackType) {
                const emailId = document.getElementById('current-email-id').value;
                
                // Validate that we have an email ID
                if (!emailId) {
                    document.getElementById('feedback-message').innerText = 'Please generate an email first';
                    return;
                }
                
                // Show feedback is being submitted
                document.getElementById('feedback-message').innerText = 'Submitting...';
                
                // Disable buttons during submission
                document.getElementById('thumbs-up-button').disabled = true;
                document.getElementById('thumbs-down-button').disabled = true;
                
                fetch('/submit-feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        feedback: feedbackType,
                        emailId: emailId
                    })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    // Show success message
                    document.getElementById('feedback-message').innerText = data.message;
                    
                    // Clear message after some time
                    setTimeout(function() {
                        document.getElementById('feedback-message').innerText = '';
                    }, 3000);
                })
                .catch(error => {
                    console.error('Error submitting feedback:', error);
                    document.getElementById('feedback-message').innerText = 'Error submitting feedback';
                    
                    // Re-enable buttons on error
                    document.getElementById('thumbs-up-button').disabled = false;
                    document.getElementById('thumbs-down-button').disabled = false;
                });
            }
        });
        """)
        
        return Title("Telecom Churn Prevention Email Generator"), Main(
            form_script,
            Div(
                H1("Telecom Churn Prevention Email Generator", cls="text-2xl font-bold text-center mb-6 text-purple-700"),
                P("Customize the customer profile to generate a personalized retention email.", 
                  cls="text-center mb-8"),
                Div(
                    controls_panel,
                    email_preview,
                    cls="flex flex-col md:flex-row gap-6 w-full"
                ),
                cls="container mx-auto px-4 py-8 max-w-6xl"
            ),
            cls="min-h-screen bg-custom-dark",
            # Add data-theme attribute to apply the custom theme
            data_theme="dark"
        )
    
    #################################################
    # Generate Email API Endpoint
    #################################################
    @rt("/generate-email", methods=["POST"])
    async def api_generate_email(request):
        """API endpoint to generate email from customer profile"""
        try:
            # Get customer profile from request JSON
            customer_profile = await request.json()
            
            # Generate email using LLM (no fallback)
            email_content = generate_email_llm.remote(customer_profile)
            
            # Get the email ID from the database (most recent)
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM emails ORDER BY created_at DESC LIMIT 1")
            result = cursor.fetchone()
            email_id = result[0] if result else None
            conn.close()
            
            # Return both the email content and ID
            response_data = {
                "email_content": email_content,
                "email_id": email_id
            }
            
            return JSONResponse(response_data)
                
        except Exception as e:
            print(f"Error generating email: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    #################################################
    # Submit Feedback API Endpoint
    #################################################
    @rt("/submit-feedback", methods=["POST"])
    async def submit_feedback(request):
        """API endpoint to submit feedback for generated emails"""
        try:
            # Get feedback data from request
            data = await request.json()
            feedback = data.get("feedback")
            email_id = data.get("emailId")  # Get email ID from request
            
            print(f"Received feedback request: {feedback} for email {email_id}")
            
            # Validate input
            if not feedback or feedback not in ["positive", "negative"]:
                return JSONResponse({"success": False, "message": "Invalid feedback value"}, status_code=400)
            
            # If no email ID provided, get the latest
            if not email_id:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM emails ORDER BY created_at DESC LIMIT 1")
                result = cursor.fetchone()
                conn.close()
                
                if not result:
                    return JSONResponse({"success": False, "message": "No email found to attach feedback"}, status_code=404)
                
                email_id = result[0]
            
            # Update the email record with feedback
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if email exists
            cursor.execute("SELECT id FROM emails WHERE id = ?", (email_id,))
            email_exists = cursor.fetchone()
            
            if not email_exists:
                conn.close()
                return JSONResponse({"success": False, "message": "Email not found"}, status_code=404)
            
            # Update feedback
            cursor.execute(
                "UPDATE emails SET feedback = ? WHERE id = ?",
                (feedback, email_id)
            )
            
            conn.commit()
            conn.close()
            
            # Log success
            print(f"✅ Saved feedback '{feedback}' for email ID: {email_id}")
            
            # Return success response with message
            feedback_message = "Thanks for your positive feedback! 👍" if feedback == "positive" else "Thanks for your feedback. We'll improve! 👎"
            return JSONResponse({
                "success": True, 
                "message": feedback_message,
                "email_id": email_id
            })
            
        except Exception as e:
            print(f"⚠️ Error submitting feedback: {e}")
            return JSONResponse({"success": False, "message": f"Error: {str(e)}"}, status_code=500)
    
    # Return the FastHTML app
    return fasthtml_app

if __name__ == "__main__":
    print("Starting Telecom Churn Prevention Email Generator...")
