import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- Configuration ---
st.set_page_config(
    page_title="Digital Marketing Conversion Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Elegant Styling ---
st.markdown("""
    <style>
        /* Global font & background */
        /* Set the main app background to a dark gradient */
        .stApp {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: #fff;
        }
        
        /* Set all headers (H1, H2, H3, H4) to a bright, contrasting color */
        h1, h2, h3, h4 {
            color: #f8f9fa !important;
        }
        
        /* Style the primary button with an attractive orange/yellow gradient */
        .stButton > button {
            background: linear-gradient(90deg, #ffb347, #ffcc33);
            color: black;
            border-radius: 12px;
            font-weight: bold;
            padding: 0.6em 1.2em;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #ffcc33, #ffb347);
            color: #222;
            box-shadow: 6px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Style st.metric boxes for results */
        .stMetric {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
        }
        
        /* Set a max width for the main container for better presentation on wide screens */
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
        }

        /* Ensure text input fields harmonize with the dark background */
        .stTextInput > div > div > input, .stNumberInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
        }

        /* Ensure st.info and st.success boxes stand out */
        [data-testid="stAlert"] div {
            border-radius: 8px;
        }

    </style>
""", unsafe_allow_html=True)


# --- File Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Gradient Boosting model."""
    try:
        # NOTE: 'gb_model.joblib' must be accessible in the environment.
        model = joblib.load('gb_model.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'gb_model.joblib' is correctly uploaded.")
        return None

# Load the model once
gb_model = load_model()

# --- Feature Definitions (Full Set for Internal Use) ---
# These lists MUST contain ALL features the model was trained on.

ORIGINAL_NUMERICAL_FEATURES = [
    'Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
    'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares',
    'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints'
]

ORIGINAL_CATEGORICAL_FEATURES = {
    'Gender': ['Female', 'Male', 'Other'],
    'CampaignChannel': ['Social Media', 'Email', 'PPC', 'Referral', 'SEO', 'Affiliate'],
    'CampaignType': ['Awareness', 'Retention', 'Acquisition'],
    'AdvertisingPlatform': ['IsConfid', 'Facebook', 'Google', 'Instagram', 'TikTok'],
    'AdvertisingTool': ['ToolConfid', 'CRM Tool', 'Automation Tool', 'Analytics Tool']
}

# --- Selected Important Features for User Input (The Simple 10) ---

SELECTED_NUMERICAL_FEATURES = [
    'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
    'TimeOnSite', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints'
]

SELECTED_CATEGORICAL_FEATURES = {
    'CampaignChannel': ORIGINAL_CATEGORICAL_FEATURES['CampaignChannel'],
    'CampaignType': ORIGINAL_CATEGORICAL_FEATURES['CampaignType']
}

# --- Preprocessing Function (CRITICAL FOR ACCURACY) ---
def preprocess_input(input_data: pd.DataFrame):
    """
    Transforms the 10 user inputs into the 24+ numerical features (including OHE) 
    required by the trained model. This fix resolves the string-to-float error 
    by ensuring only numerical columns are passed to the model while also 
    satisfying the model's requirement to see specific raw categorical column names.
    """
    
    # 1. Fill missing (unselected) features with consistent defaults
    
    # 1a. Fill unselected numerical features (defaulting to 0)
    for col in ORIGINAL_NUMERICAL_FEATURES:
        if col not in input_data.columns:
            input_data[col] = 0.0
            
    # 1b. Fill unselected categorical features (defaulting to the first category)
    for feature, categories in ORIGINAL_CATEGORICAL_FEATURES.items():
        if feature not in input_data.columns:
            # We ensure the default value is a known category for OHE
            input_data[feature] = categories[0]
            
    # 2. Apply One-Hot Encoding (OHE)
    # This automatically drops the raw categorical columns and keeps the numerical columns.
    processed_df = pd.get_dummies(
        input_data, 
        prefix=list(ORIGINAL_CATEGORICAL_FEATURES.keys()), 
        columns=list(ORIGINAL_CATEGORICAL_FEATURES.keys())
    )

    # 3. Handle missing dummy columns (guaranteeing all OHE columns exist)
    expected_dummy_columns = []
    for feature, categories in ORIGINAL_CATEGORICAL_FEATURES.items():
        for category in categories:
            expected_dummy_columns.append(f'{feature}_{category}')

    # Add missing OHE columns with a value of 0
    for col in expected_dummy_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0 
    
    # --- FIX FOR FEATURE NAME ERROR (V2) ---
    if hasattr(gb_model, 'feature_names_in_') and gb_model.feature_names_in_ is not None:
        
        # 4. Check for and add back any expected RAW categorical columns (as numerical placeholders)
        # This addresses the "missing features" error while keeping the data numerical.
        
        # Identify raw categorical column names that the model's feature list contains
        raw_cat_cols_in_model = [
            col for col in list(gb_model.feature_names_in_) 
            if col in ORIGINAL_CATEGORICAL_FEATURES.keys()
        ]
        
        for col in raw_cat_cols_in_model:
            # If the raw string column is in the model's expected list, 
            # but was dropped by pd.get_dummies, add it back as a column of zeros 
            # to satisfy the feature name check.
            if col not in processed_df.columns:
                processed_df[col] = 0.0 
        
        try:
            # 5. Filter and enforce the exact column order the model expects (MOST CRITICAL STEP)
            # This DataFrame now contains numerical, OHE, and numerical placeholder columns, 
            # matching the model's exact feature_names_in_ list.
            final_input_df = processed_df[list(gb_model.feature_names_in_)]
            
        except KeyError as e:
            st.error(f"Internal Feature Error: Column mismatch in final numerical/OHE set detected ({e}).")
            return None
    else:
        # Fallback (less robust)
        final_input_df = processed_df
    
    return final_input_df


# --- Insight Generation Function (Simulated Explainability) ---
def generate_insights(raw_data):
    """Generates simple, rule-based insights based on input values, offering an explanation."""
    
    # Extract first (and only) row of data
    data = {k: v[0] for k, v in raw_data.items()}
    
    positive_drivers = []
    negative_drivers = []
    
    # Thresholds (assumed from typical business logic for clear classification)
    THRESHOLDS = {
        'Income_High': 100000, 'Income_Low': 50000,
        'AdSpend_High': 1000, 'AdSpend_Low': 100,
        'ClickThroughRate_High': 0.25, 'ClickThroughRate_Low': 0.05,
        'TimeOnSite_High': 30.0, 'TimeOnSite_Low': 5.0,
        'EmailClicks_High': 5, 'EmailClicks_Low': 1,
        'PreviousPurchases_High': 10, 'PreviousPurchases_Low': 1,
        'LoyaltyPoints_High': 3000, 'LoyaltyPoints_Low': 500
    }
    
    # --- Numerical Analysis ---
    if data['Income'] > THRESHOLDS['Income_High']:
        positive_drivers.append(f"High Income ($\${data['Income']:.0f}$) suggests strong purchasing power.")
    elif data['Income'] < THRESHOLDS['Income_Low']:
        negative_drivers.append(f"Low Income ($\${data['Income']:.0f}$) may limit immediate conversion.")

    if data['AdSpend'] > THRESHOLDS['AdSpend_High']:
        positive_drivers.append(f"High Ad Spend ($\${data['AdSpend']:.0f}$) indicates high segment value.")
    
    if data['ClickThroughRate'] > THRESHOLDS['ClickThroughRate_High']:
        positive_drivers.append(f"High CTR ({data['ClickThroughRate']:.2%}) indicates strong ad relevance.")
    elif data['ClickThroughRate'] < THRESHOLDS['ClickThroughRate_Low']:
        negative_drivers.append(f"Low CTR ({data['ClickThroughRate']:.2%}) suggests weak interest in the creative.")

    if data['TimeOnSite'] > THRESHOLDS['TimeOnSite_High']:
        positive_drivers.append(f"High Time On Site ({data['TimeOnSite']:.0f}s) shows deep content engagement.")
    elif data['TimeOnSite'] < THRESHOLDS['TimeOnSite_Low']:
        negative_drivers.append(f"Very Low Time On Site ({data['TimeOnSite']:.0f}s) suggests a bounce or poor experience.")

    if data['EmailClicks'] > THRESHOLDS['EmailClicks_High']:
        positive_drivers.append(f"High Email Clicks ({data['EmailClicks']}) signal direct, high-value intent.")
    
    if data['PreviousPurchases'] > THRESHOLDS['PreviousPurchases_High']:
        positive_drivers.append(f"High Previous Purchases ({data['PreviousPurchases']}) indicates an established, loyal customer.")
    elif data['PreviousPurchases'] <= THRESHOLDS['PreviousPurchases_Low']:
        negative_drivers.append(f"Few Previous Purchases ({data['PreviousPurchases']}) suggests limited purchase history.")

    if data['LoyaltyPoints'] > THRESHOLDS['LoyaltyPoints_High']:
        positive_drivers.append(f"High Loyalty Points ({data['LoyaltyPoints']}) demonstrate strong brand commitment.")
    elif data['LoyaltyPoints'] < THRESHOLDS['LoyaltyPoints_Low']:
        negative_drivers.append(f"Low Loyalty Points ({data['LoyaltyPoints']}) suggests low engagement or a new customer.")
        
    # --- Categorical Analysis ---
    if data['CampaignType'] == 'Acquisition':
        positive_drivers.append("Campaign Type: Acquisition - profile matches target for new customer conversion.")
    elif data['CampaignType'] == 'Awareness':
        negative_drivers.append("Campaign Type: Awareness - conversion is generally lower for top-of-funnel goals.")

    if data['CampaignChannel'] in ['Email', 'PPC']:
        positive_drivers.append(f"Channel: {data['CampaignChannel']} - these are typically high-intent, lower-funnel channels.")
    
    return {'positive': positive_drivers, 'negative': negative_drivers}


# --- Streamlit UI Implementation ---

def main():
    """Main function to run the Streamlit application."""
    # Updated title to leverage the dark background better
    st.title("🎯 Digital Marketing Conversion Predictor")
    st.markdown("---")
    
    # Project goal/model performance info block removed here
    st.markdown("""
        Enter the **10 key campaign metrics** below to get an accurate, data-driven conversion prediction.
    """)
    

    if gb_model is None:
        return

    # Use a Streamlit Form for clean, atomic submission (best practice)
    with st.form(key='conversion_form', clear_on_submit=False):
        
        # --- INPUT SECTION (Simplified for User) ---
        
        # Row 1: Campaign Strategy (Categorical)
        st.subheader("Campaign Strategy & Channels")
        col_strategy_1, col_strategy_2 = st.columns(2)
        with col_strategy_1:
            input_channel = st.selectbox("Campaign Channel (Where Customer Was Reached)", SELECTED_CATEGORICAL_FEATURES['CampaignChannel'])
        with col_strategy_2:
            input_type = st.selectbox("Campaign Goal (Type of Campaign)", SELECTED_CATEGORICAL_FEATURES['CampaignType'])

        st.markdown("---")
        
        # Row 2: Financials and Rates (Monetary and Percentage Sliders)
        st.subheader("Financial & Efficiency Metrics")
        col_financial_1, col_financial_2, col_financial_3, col_financial_4 = st.columns(4)

        with col_financial_1:
            input_income = st.number_input("Customer Income ($)", min_value=10000, max_value=250000, value=75000, step=1000, help="Estimate of customer purchasing power.")
        with col_financial_2:
            input_adspend = st.number_input("Ad Spend ($)", min_value=0.0, max_value=10000.0, value=500.0, step=50.0, help="Investment level in reaching this segment.")
        with col_financial_3:
            # Using slider for visual input of percentage rate
            input_ctr = st.slider("Click Through Rate (CTR)", min_value=0.0, max_value=1.0, value=0.15, step=0.01, format="%.2f", help="Measures interest in the ad (0.0 to 1.0).")
        with col_financial_4:
            # Using slider for visual input of percentage rate
            input_conversion_rate = st.slider("Historical Conversion Rate", min_value=0.0, max_value=1.0, value=0.08, step=0.01, format="%.2f", help="Past success rate for this segment (0.0 to 1.0).")

        st.markdown("---")

        # Row 3: Engagement and Loyalty (Counts)
        st.subheader("Engagement")
        col_engagement_1, col_engagement_2, col_engagement_3, col_engagement_4 = st.columns(4)
        
        with col_engagement_1:
            input_time_on_site = st.number_input("Time On Site (Seconds)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, help="Average time spent on the website.")
        with col_engagement_2:
            input_email_clicks = st.number_input("Email Clicks (Total Count)", min_value=0, max_value=50, value=2, step=1, help="Strong signal of intent from email marketing.")
        with col_engagement_3:
            input_prev_purchases = st.number_input("Previous Purchases (Count)", min_value=0, max_value=50, value=5, step=1, help="Number of items or transactions in history.")
        with col_engagement_4:
            input_loyalty = st.number_input("Loyalty Points", min_value=0, max_value=10000, value=1500, step=100, help="Customer loyalty status score.")


        # --- Prediction Button ---
        st.markdown("---")
        submitted = st.form_submit_button("🚀PREDICT CONVERSION LIKELIHOOD", type="primary", use_container_width=True)

    
    # --- Prediction Logic (executed only after form submission) ---
    if submitted:
        
        # 1. Assemble the 10 user inputs
        raw_data = {
            'Income': [input_income], 'AdSpend': [input_adspend], 'ClickThroughRate': [input_ctr], 
            'ConversionRate': [input_conversion_rate], 'TimeOnSite': [input_time_on_site], 
            'EmailClicks': [input_email_clicks], 'PreviousPurchases': [input_prev_purchases], 
            'LoyaltyPoints': [input_loyalty], 'CampaignChannel': [input_channel], 
            'CampaignType': [input_type]
        }
        input_df = pd.DataFrame(raw_data)

        # 2. Translate inputs for the model
        final_input = preprocess_input(input_df)

        if final_input is not None and not final_input.empty:
            # 3. Make Prediction
            prediction_proba = gb_model.predict_proba(final_input)[:, 1][0] # Probability of Conversion=1
            prediction_class = gb_model.predict(final_input)[0]

            # 4. Display Results
            st.subheader("🎯Prediction Outcome")
            
            col_res_1, col_res_2 = st.columns(2)

            with col_res_1:
                if prediction_class == 1:
                    st.success("✅CONVERSION IS HIGHLY LIKELY")
                    # Direct action based on project goals (Improved Targeting)
                    st.markdown(f"**Action:** Prioritize this segment for immediate, high-value outreach to maximize ROAS!")
                    st.balloons()
                else:
                    st.error("CONVERSION IS UNLIKELY")
                    # Direct action based on project goals (Cost Optimization)
                    st.markdown(f"**Action:** Reallocate budget from this low-likelihood segment, consider a different campaign type.")
            
            with col_res_2:
                st.metric(label="Probability of Conversion (Target=1)", value=f"{prediction_proba:.2%}")

            # 5. Display Dynamic Insights (Simplified Section for Explainability)
            st.markdown("---")
            st.subheader("💡 Key Conversion Drivers for this Profile")
            # Removed explanatory markdown: st.markdown("Understanding *why* the prediction was made helps optimize strategy:")
            
            insights = generate_insights(raw_data)
            
            col_insight_1, col_insight_2 = st.columns(2)

            with col_insight_1:
                st.markdown("**Positive Influences**:") # Simplified title
                if insights['positive']:
                    for p in insights['positive']:
                        st.success(f"- {p}")
                else:
                    st.info("- No strong positive drivers identified.") # Simplified message
                    
            with col_insight_2:
                st.markdown("**Negative Influences**:") # Simplified title
                if insights['negative']:
                    for n in insights['negative']:
                        st.warning(f"- {n}")
                else:
                    st.info("- No strong negative drivers identified.") # Simplified message

            # Optional: Show the input data used for prediction
            with st.expander("Model Verification: View the Full Feature Set Input"):
                st.dataframe(final_input)


if __name__ == "__main__":
    main()
