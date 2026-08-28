import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE  


st.set_page_config(page_title="eBay Risk Gateway (Corrected Balance)", layout="wide")

st.title("Credit Card Fraud Detection")
st.caption("Ekansh Kumar Rajpoot")

@st.cache_data
def process_production_pipeline():
    
    df = pd.read_csv('creditcard.csv')
    X = df.drop(columns='Class')
    Y = df['Class']
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    
    
    smote = SMOTE(random_state=42)
    X_train_bal, Y_train_bal = smote.fit_resample(X_train, Y_train)
    
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_bal, Y_train_bal)
    
    return model, X_train_bal, Y_train_bal, X_test, Y_test

try:
    
    model, X_train_bal, Y_train_bal, X_test, Y_test = process_production_pipeline()
    
    
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, y_pred)
    
    
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    
    
    st.subheader("Pipeline Balance Verification")
    col_bal1, col_bal2 = st.columns(2)
    
    
    col_bal1.metric("Balanced Training Rows Set", f"{len(Y_train_bal):,}", "Exact 50:50 Class Ratio")

    col_bal2.metric("True Testing Accuracy Score", f"{test_accuracy * 100:.2f}%", "Leakage-Free Validation")
    
    st.markdown("---")
    
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Unseen Test Cases", f"{len(Y_test):,}")
    m2.metric("Actual Fraud Caught (True Positives)", f"{tp:,}")
    m3.metric("Legitimate Users Blocked (False Positives)", f"{fp:,}")
    
    st.markdown("---")
    
    
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("Model Interpretation Breakdown")
        st.markdown(f"""
        Evaluating core error classifications after neutralizing structural sampling leakage:
        * **True Negatives** (Valid clear runs correctly approved): `{tn:,}`
        * **False Positives** (Accidentally flagged standard users): `{fp:,}`
        * **False Negatives** (Undetected leakage cases / Chargeback leaks): `{fn:,}`
        * **True Positives** (Fraud patterns successfully stopped): `{tp:,}`
        
        *Note: Balancing our optimization space allows the classifier to catch significantly more fraud alerts, driving down catastrophic `False Negatives`.*
        """)
        
    with right_col:
        st.subheader("Real-Time Transaction Inspector Portal")
        with st.form("triage_form"):
            v1 = st.number_input("PCA Structural Variable V1", value=0.0)
            v2 = st.number_input("PCA Structural Variable V2", value=0.0)
            v3 = st.number_input("PCA Structural Variable V3", value=0.0)
            amount = st.number_input("Transaction Purchase Amount Volume ($)", value=50.0)
            
            submit = st.form_submit_button("Run Risk Assessment Engine")
            
        if submit:
            
            mock_features = [v1, v2, v3] + [0.0]*26 + [amount]
            input_df = pd.DataFrame([mock_features], columns=X_test.columns)
            
            
            risk_prob = model.predict_proba(input_df)[0][1]
            
            st.write(f"**Computed Fraud Risk Probability Density:** `{risk_prob * 100:.2f}%`")
            if risk_prob > 0.5:
                st.error("MITIGATION TRIGGERED: Route Transaction directly to Fraud Review Queue")
            else:
                st.success("SETTLEMENT APPROVED: Forward Payload directly to Payment Clearing Network")

except FileNotFoundError:
    st.error("Please place the target 'creditcard.csv' payload within this project directory path footprint.")
