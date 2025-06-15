import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# @st.cache_resource
def load_model():
    return joblib.load("model/fraud_detector_pipeline.pkl")

model = load_model()

#Page setup
st.set_page_config(page_title="Fake Job Detector", layout="wide")
st.title("🛡️ Job Posting Fraud Detector")


st.markdown("""
Upload a CSV file containing job listings.  
The model will predict whether each listing is **fraudulent** or **genuine**, along with probability.
""")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    #preprocessing the uploaded file
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    num_cols = ['telecommuting', 'has_company_logo', 'has_questions']

    for col in text_cols + cat_cols + num_cols:
        if col not in data.columns:
            data[col] = np.nan if col in cat_cols else 0
    
    data['text'] = data[text_cols].fillna("").agg(" ".join, axis = 1)
    X_input = data[['text'] + cat_cols + num_cols]

    #predicting probabiliy
    pred_probs = model.predict_proba(X_input)[:, 1]
    preds = model.predict(X_input)

    #storing the data to data frame
    results_df = data.copy()
    results_df['Fraud Probability'] = pred_probs
    results_df['Prediction'] = preds
    results_df['Prediction Label'] = results_df['Prediction'].map({0: 'Genuine', 1: 'Fraudulent'})

    #display full prediciton table
    st.subheader("Prediction Table")
    st.dataframe(results_df[['title', 'location', 'Prediction Label', 'Fraud Probability']].sort_values(by = 'Fraud Probability', ascending=False))

    #Display Graphyical representation
    
    #1 histogram
    col1, col2 = st.columns(2)
    with col1: 
        st.subheader("Histogram Fraud Probability")
        fig1, ax1 = plt.subplots(figsize = (5,3.7))
        sns.histplot(results_df['Fraud Probability'], bins = 20, kde=True, ax = ax1, color='red')
        ax1.set_xlabel("Fraud Probability")
        ax1.set_ylabel("Number of Job Listings")
        st.pyplot(fig1)

    #plot 2: pie chart - Real VS Fake
    with col2:
        st.subheader("Prediction Breakdown")
        pie_counts = results_df['Prediction Label'].value_counts()
        fig2, ax2 = plt.subplots(figsize = (5,4))
        ax2.pie(pie_counts, labels = pie_counts.index, autopct = '%1.1f%%', colors = ['green', 'red'])
        ax2.axis('equal')
        st.pyplot(fig2)


    #Top 10 most suspicios listings
    st.subheader("Top 10 most suspicious Listings")
    top10 = results_df.sort_values(by = 'Fraud Probability', ascending=False).head(10)
    st.dataframe(top10[['title', 'location', 'description', 'Fraud Probability']])

else:
    st.info("Please Upload a CSV file to get started.")