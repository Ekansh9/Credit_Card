import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="eBay Risk Gateway (Corrected Balance)", layout="wide")

st.title("Credit Card Fraud Detection")
st.caption("Ekansh Kumar Rajpoot")

@st.cache_data
def process_production_pipeline():
    df = pd.read_csv('creditcard.csv')
    
    # 1. Time aur Class drop 
    X = df.drop(columns=['Class', 'Time'])  
    Y = df['Class']
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    
    # 2. Scaling added
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 3. SMOTE on scaled data
    smote = SMOTE(random_state=42)
    X_train_bal, Y_train_bal = smote.fit_resample(X_train_scaled, Y_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_bal, Y_train_bal)

    return model, scaler, X_train_bal, Y_train_bal, X_test_scaled, Y_test, X_test

try:
    #4. Function call update  (scaler, X_test add )
    model, scaler, X_train_bal, Y_train_bal, X_test_scaled, Y_test, X_test_original = process_production_pipeline()
    
    #5. Prediction on scaled data 
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(Y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    
    test_precision = precision_score(Y_test, y_pred)
    test_recall    = recall_score(Y_test, y_pred)
    test_f1        = f1_score(Y_test, y_pred)
    test_roc_auc   = roc_auc_score(Y_test, model.predict_proba(X_test_scaled)[:, 1])

   
    
    #  6. Class Distribution 
    st.subheader("Original Class Distribution")
    st.bar_chart(pd.DataFrame({'Count': [284315, 492]}, index=['Normal', 'Fraud']))
    st.markdown("---")

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
    
    # 7. Quality Metrics 
    st.subheader("Classification Quality Metrics")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Precision", f"{test_precision*100:.2f}%")
    q2.metric("Recall",    f"{test_recall*100:.2f}%")
    q3.metric("F1 Score",  f"{test_f1*100:.2f}%")
    q4.metric("ROC-AUC",   f"{test_roc_auc:.4f}")
    
    st.markdown("---")
    
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("Model Interpretation Breakdown")
        st.markdown(f"""
        * **True Negatives** (Valid clear runs correctly approved): `{tn:,}`
        * **False Positives** (Accidentally flagged standard users): `{fp:,}`
        * **False Negatives** (Undetected leakage cases / Chargeback leaks): `{fn:,}`
        * **True Positives** (Fraud patterns successfully stopped): `{tp:,}`
        """)
        
        # 8. Confusion Matrix Heatmap 
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Normal', 'Pred Fraud'],
                    yticklabels=['Actual Normal', 'Actual Fraud'], ax=ax)
        st.pyplot(fig)
        
        # 9. Top Fraud Indicators 
        st.subheader("Top Fraud Indicators (Model Coefficients)")
        coef_df = pd.DataFrame({
            'Feature': X_test_original.columns, # Note: X_test_original use kiya kyunki Time drop ho chuka hai
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False).head(10)
        st.bar_chart(coef_df.set_index('Feature'))
        
    with right_col:
        st.subheader("Real-Time Transaction Inspector Portal")
        with st.form("triage_form"):
            v1 = st.number_input("PCA Structural Variable V1", value=0.0)
            v2 = st.number_input("PCA Structural Variable V2", value=0.0)
            v3 = st.number_input("PCA Structural Variable V3", value=0.0)
            amount = st.number_input("Transaction Purchase Amount Volume ($)", value=50.0)
            
            # 10. Threshold Slider 
            threshold = st.slider("Decision Threshold", 0.1, 0.99, 0.90, 0.01)
            
            submit = st.form_submit_button("Run Risk Assessment Engine")
            
        if submit:
            mock_features = [v1, v2, v3] + [0.0]*25 + [amount]
            input_df = pd.DataFrame([mock_features], columns=X_test_original.columns)
            
            # 12. Input scale 
            input_scaled = scaler.transform(input_df)
            
            # 13.  probability from Scaled input 
            risk_prob = model.predict_proba(input_scaled)[0][1]
            
            st.write(f"**Computed Fraud Risk Probability Density:** `{risk_prob * 100:.2f}%`")
            
            # 14. Slider threshold ka use karo
            if risk_prob > threshold:
                st.error("MITIGATION TRIGGERED: Route Transaction directly to Fraud Review Queue")
            else:
                st.success("SETTLEMENT APPROVED: Forward Payload directly to Payment Clearing Network")

except FileNotFoundError:
    st.error("Please place the target 'creditcard.csv' payload within this project directory path footprint.")
    