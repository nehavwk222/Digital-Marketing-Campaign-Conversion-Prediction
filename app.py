import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Digital Marketing Conversion Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: #fff;
}
h1, h2, h3, h4 {
    color: #f8f9fa !important;
}
.stButton > button {
    background: linear-gradient(90deg, #ffb347, #ffcc33);
    color: black;
    border-radius: 12px;
    font-weight: bold;
    padding: 0.6em 1.2em;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #ffcc33, #ffb347);
    color: #222;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}
.stTextInput > div > div > input, .stNumberInput > div > div > input {
    background-color: rgba(255, 255, 255, 0.1);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 5px;
}
[data-testid="stAlert"] div {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --- Load Pipeline ---
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load('conversion_pipeline.joblib')
        return pipeline
    except Exception as e:
        st.error(f"Error loading pipeline: {e}. Ensure 'conversion_pipeline.joblib' is uploaded.")
        return None

pipeline = load_pipeline()

# --- Feature Lists ---
NUM_FEATURES = [
    'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
    'TimeOnSite', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints'
]

CATEGORICAL_FEATURES = {
    'CampaignChannel': ['Social Media', 'Email', 'PPC', 'Referral', 'SEO', 'Affiliate'],
    'CampaignType': ['Awareness', 'Retention', 'Acquisition']
}

# --- Preprocess Input for Pipeline ---
def preprocess_input(input_df: pd.DataFrame):
    for col in NUM_FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0.0
    for feature, categories in CATEGORICAL_FEATURES.items():
        if feature not in input_df.columns:
            input_df[feature] = categories[0]

    df_processed = pd.get_dummies(input_df)
    if hasattr(pipeline, 'feature_names_in_') and pipeline.feature_names_in_ is not None:
        for col in pipeline.feature_names_in_:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[list(pipeline.feature_names_in_)]
    return df_processed

# --- Generate Simple Insights ---
def generate_insights(raw_data):
    data = {k: v[0] for k, v in raw_data.items()}
    positive, negative = [], []

    if data['Income'] > 100000: positive.append("High Income: strong purchasing power")
    elif data['Income'] < 50000: negative.append("Low Income: may limit conversion")
    
    if data['AdSpend'] > 1000: positive.append("High Ad Spend: valuable segment")
    
    if data['ClickThroughRate'] > 0.25: positive.append("High CTR: strong ad relevance")
    elif data['ClickThroughRate'] < 0.05: negative.append("Low CTR: weak engagement")
    
    if data['TimeOnSite'] > 30: positive.append("High Time On Site: deep engagement")
    elif data['TimeOnSite'] < 5: negative.append("Low Time On Site: possible bounce")
    
    if data['EmailClicks'] > 5: positive.append("High Email Clicks: strong intent")
    if data['PreviousPurchases'] > 10: positive.append("High Previous Purchases: loyal customer")
    elif data['PreviousPurchases'] <= 1: negative.append("Few Previous Purchases: low history")
    
    if data['LoyaltyPoints'] > 3000: positive.append("High Loyalty Points: strong brand commitment")
    elif data['LoyaltyPoints'] < 500: negative.append("Low Loyalty Points: new or disengaged customer")
    
    if data['CampaignType'] == 'Acquisition': positive.append("Acquisition Campaign: good for new conversions")
    elif data['CampaignType'] == 'Awareness': negative.append("Awareness Campaign: usually low conversions")
    
    if data['CampaignChannel'] in ['Email', 'PPC']: positive.append(f"{data['CampaignChannel']} channel: high-intent traffic")
    
    return {'positive': positive, 'negative': negative}

# --- Streamlit UI ---
def main():
    st.title("🎯 Digital Marketing Conversion Predictor")
    st.markdown("Enter the **10 key metrics** below to get a conversion prediction.")

    if pipeline is None: 
        return

    with st.form("conversion_form", clear_on_submit=False):
        st.subheader("Campaign Strategy & Channels")
        col1, col2 = st.columns(2)
        with col1:
            channel = st.selectbox("Campaign Channel", CATEGORICAL_FEATURES['CampaignChannel'])
        with col2:
            ctype = st.selectbox("Campaign Type", CATEGORICAL_FEATURES['CampaignType'])

        st.subheader("Financial & Efficiency Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            income = st.number_input("Income ($)", min_value=10000, max_value=250000, value=75000, step=1000)
        with col2:
            adspend = st.number_input("Ad Spend ($)", min_value=0, max_value=10000, value=500, step=50)
        with col3:
            ctr = st.slider("Click Through Rate (CTR)", 0.0, 1.0, 0.15, 0.01, format="%.2f")
        with col4:
            conv_rate = st.slider("Historical Conversion Rate", 0.0, 1.0, 0.08, 0.01, format="%.2f")

        st.subheader("Engagement & Loyalty Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            time_site = st.number_input("Time On Site (Seconds)", 0.0, 100.0, 15.0, 1.0)
        with col2:
            email_clicks = st.number_input("Email Clicks", 0, 50, 2, 1)
        with col3:
            prev_purchases = st.number_input("Previous Purchases", 0, 50, 5, 1)
        with col4:
            loyalty = st.number_input("Loyalty Points", 0, 10000, 1500, 100)

        submitted = st.form_submit_button("🚀 PREDICT CONVERSION")

    if submitted:
        raw_input = {
            'Income':[income], 'AdSpend':[adspend], 'ClickThroughRate':[ctr],
            'ConversionRate':[conv_rate], 'TimeOnSite':[time_site],
            'EmailClicks':[email_clicks], 'PreviousPurchases':[prev_purchases],
            'LoyaltyPoints':[loyalty], 'CampaignChannel':[channel],
            'CampaignType':[ctype]
        }

        input_df = pd.DataFrame(raw_input)
        final_input = preprocess_input(input_df)

        if final_input is not None and not final_input.empty:
            proba = pipeline.predict_proba(final_input)[:,1][0]

            # --- Improved Probability Interpretation ---
            st.subheader("🎯 Prediction Result")
            col1, col2 = st.columns([2, 1])
            with col1:
                if proba >= 0.8:
                    st.success("✅ CONVERSION IS HIGHLY LIKELY")
                    st.markdown("**Action:** Prioritize this segment for immediate high-value outreach!")
                    st.balloons()
                elif proba >= 0.5:
                    st.warning("⚠️ CONVERSION MODERATELY LIKELY")
                    st.markdown("**Action:** Engage further — test higher-value offers or retargeting.")
                else:
                    st.error("❌ CONVERSION UNLIKELY")
                    st.markdown("**Action:** Consider reallocating budget or refining audience targeting.")
            with col2:
                st.metric("Probability of Conversion", f"{proba:.2%}")

            # --- Visual Probability Bar ---
            st.markdown("### 📊 Conversion Probability Indicator")
            progress_value = float(proba)
            progress_color = "green" if proba >= 0.8 else "orange" if proba >= 0.5 else "red"
            st.progress(progress_value)

            # --- Key Conversion Drivers ---
            st.subheader("💡 Key Conversion Drivers")
            insights = generate_insights(raw_input)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Positive Drivers:**")
                if insights['positive']:
                    for p in insights['positive']: st.success(f"- {p}")
                else: st.info("- No strong positive drivers")
            with col2:
                st.markdown("**Negative Drivers:**")
                if insights['negative']:
                    for n in insights['negative']: st.warning(f"- {n}")
                else: st.info("- No strong negative drivers")

            with st.expander("View Full Feature Set Input"):
                st.dataframe(final_input)

if __name__ == "__main__":
    main()
