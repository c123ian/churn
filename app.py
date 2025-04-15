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

# Generation prompt template
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
    email_content = output[0]["generated_text"][len(prompt):].strip()
    
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
    
    return email_content

# Fallback email generator (non-LLM)
def generate_email_content(customer_profile: Dict[str, Any]) -> str:
    """Generate email content based on customer profile (non-LLM fallback)"""
    # Extract customer name
    customer_name = customer_profile.get("customerName", "Valued Customer")
    
    # Determine discount percentage based on monthly charges
    monthly_charges = customer_profile.get("monthlyCharges", 0)
    discount_percentage = "15%"
    if monthly_charges > 75:
        discount_percentage = "30%"
    elif monthly_charges > 50:
        discount_percentage = "25%"
    elif monthly_charges > 25:
        discount_percentage = "20%"
    
    # Create subject line based on key factors
    if customer_profile.get("monthToMonth", False) and monthly_charges > 75:
        subject = f"Exclusive offer: Save {discount_percentage} on your monthly bill"
    elif customer_profile.get("noOnlineSecurity", False):
        subject = "Protect your digital life with our premium security package"
    elif customer_profile.get("lowTenure", False):
        subject = f"Welcome to the family, {customer_name}! Special offers inside"
    else:
        subject = "Special savings tailored just for you"
    
    # Create personalized greeting
    greeting = f"Dear {customer_name},"
    
    # Create introduction based on tenure and spending
    if customer_profile.get("lowTenure", False):
        intro = "We're thrilled to have you as part of our Vodafone family and want to ensure you're getting the most from your service."
    elif monthly_charges > 75:
        intro = f"As one of our premium customers, we value your loyalty and would like to thank you with some exclusive offers that could significantly reduce your monthly spend of ${monthly_charges}."
    else:
        intro = "We appreciate your continued loyalty and have put together some special offers tailored specifically to enhance your experience."
    
    # Create benefits based on active factors (limit to 3-4)
    benefits = []
    
    if customer_profile.get("monthToMonth", False) and monthly_charges > 50:
        discounted_amount = round(monthly_charges * (1 - int(discount_percentage.strip('%')) / 100))
        benefits.append(f"Switch from your month-to-month plan to our annual commitment and receive a {discount_percentage} discount, bringing your monthly bill down to ${discounted_amount}.")
    
    if customer_profile.get("fiberOptic", False):
        benefits.append("Upgrade to our premium router at no extra cost and receive a complimentary technical assessment to optimize your fiber connection speed and reliability.")
    
    if customer_profile.get("noOnlineSecurity", False):
        benefits.append("Protect your digital life with 6 months of our premium security package at no cost, including antivirus, anti-phishing, and identity protection services.")
    
    if customer_profile.get("noPartner", False):
        benefits.append("Bring a friend to our network and you'll both receive three months of service at half price! It's our way of saying thanks for spreading the word.")
    
    if customer_profile.get("seniorCitizen", False):
        benefits.append("As a distinguished member of our community, you're eligible for our senior package with simplified billing, dedicated tech support, and a 15% lifetime discount.")
    
    # Limit to top 3-4 benefits
    benefits = benefits[:4]
    
    # Create call to action
    if customer_profile.get("monthToMonth", False) and monthly_charges > 50:
        cta = f"To take advantage of the {discount_percentage} discount, simply reply to this email or call us at 0800-VODAFONE by May 1st."
    elif customer_profile.get("noOnlineSecurity", False):
        cta = "Activate your free security package today by visiting vodafone.com/security or calling our customer service team."
    else:
        cta = "To claim these exclusive offers, please call our dedicated customer care line at 0800-VODAFONE or visit your account dashboard."
    
    # Create closing and signature
    closing = "We value your business and look forward to continuing to serve your telecommunications needs."
    signature = "Warm regards,\n\nSarah Thompson\nCustomer Retention Team\nVodafone"
    
    # Compile the complete email
    email_content = f"Subject: {subject}\n\n{greeting}\n\n{intro}\n\nWe've prepared some exclusive offers just for you:\n"
    
    # Add benefits as bullet points
    for benefit in benefits:
        email_content += f"• {benefit}\n"
    
    # Add call to action, closing and signature
    email_content += f"\n{cta}\n\n{closing}\n\n{signature}"
    
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
            H2("Customer Profile", cls="text-xl font-bold mb-4"),
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
            cls="w-full md:w-1/2 bg-white p-6 rounded-lg shadow"
        )
        
        # Email preview panel
        email_preview = Div(
            H2("Generated Email", cls="text-xl font-bold mb-4"),
            Div(
                Div(
                    cls="loading loading-spinner loading-lg",
                    id="loading-indicator"
                ),
                cls="flex justify-center items-center h-12 hidden"
            ),
            Textarea(
                placeholder="Your generated email will appear here...",
                cls="textarea textarea-bordered w-full h-96",
                id="email-content",
                readonly="readonly"
            ),
            Div(
                Button(
                    "Copy to Clipboard",
                    cls="btn btn-outline btn-sm",
                    hx_on_click="navigator.clipboard.writeText(document.getElementById('email-content').value); this.innerText = 'Copied!'; setTimeout(() => this.innerText = 'Copy to Clipboard', 2000)"
                ),
                cls="mt-4 flex justify-end"
            ),
            cls="w-full md:w-1/2 bg-white p-6 rounded-lg shadow"
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
            
            // Only generate email when the button is clicked
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
                .then(response => response.text())
                .then(data => {
                    document.getElementById('email-content').value = data;
                    // Hide loading indicator
                    document.getElementById('loading-indicator').parentElement.classList.add('hidden');
                });
            });
            
            // Update the charges value display without triggering email generation
            document.getElementById('charges-slider').addEventListener('input', function() {
                document.getElementById('charges-value').innerText = this.value;
            });
        });
        """)
        
        return Title("Telecom Churn Prevention Email Generator"), Main(
            form_script,
            Div(
                H1("Telecom Churn Prevention Email Generator", cls="text-2xl font-bold text-center mb-6"),
                P("Customize the customer profile to generate a personalized retention email.", 
                  cls="text-center mb-8"),
                Div(
                    controls_panel,
                    email_preview,
                    cls="flex flex-col md:flex-row gap-6 w-full"
                ),
                cls="container mx-auto px-4 py-8 max-w-6xl"
            ),
            cls="min-h-screen bg-gray-100"
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
            
            try:
                # Try to use the LLM-based generator
                email_content = generate_email_llm.remote(customer_profile)
            except Exception as llm_error:
                print(f"⚠️ LLM generation failed: {llm_error}. Using fallback generator.")
                # Fallback to rule-based generator if LLM fails
                email_content = generate_email_content(customer_profile)
            
            # Return the generated email
            return HTMLResponse(email_content)
            
        except Exception as e:
            print(f"Error generating email: {e}")
            return HTMLResponse(f"Error generating email: {str(e)}", status_code=500)
    
    # Return the FastHTML app
    return fasthtml_app

if __name__ == "__main__":
    print("Starting Telecom Churn Prevention Email Generator...")
