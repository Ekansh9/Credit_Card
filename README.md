A Machine Learning pipeline built to detect fraudulent credit card transactions. Because fraud accounts for only 0.17% of the data, standard accuracy is misleading.
This project uses Logistic Regression with optimized thresholding to prioritize catching frauds (Recall) while minimizing false alarms.
Dataset Source : The dataset is the Credit Card Fraud Detection Dataset provided by the ULB (Université Libre de Bruxelles) Machine Learning Group.
Source: Kaggle - Credit Card Fraud Detection
Size: 284,807 transactions.
Fraud Cases: 492 (0.172%).
Features: 28 PCA transformed features (V1-V28), Time and Amount.
Preprocessing: Dropped Time and applied StandardScaler to the Amount feature.
Imbalance Handling: Applied SMOTE (Synthetic Minority Over sampling Technique) strictly on the training data to prevent data leakage.
Model: Trained a Logistic Regression model (max_iter=1000).
Threshold Tuning: Adjusted the decision threshold from the default 0.5 to 0.90 to drastically reduce False Negatives (missed frauds).
Evaluation Metrics: Optimized for ROC-AUC, F1-Score, and Recall, using a detailed Confusion Matrix to analyze the business cost of errors.
Streamlit App: Includes an interactive dashboard (app.py) for real time transaction triage, featuring a dynamic threshold slider and visual feature importance.
Launch the Streamlit app:
bash
streamlit run app.py
