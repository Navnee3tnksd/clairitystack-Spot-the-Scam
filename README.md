# 🛡️ Fake Job Posting Detection

Detects fraudulent job postings using machine learning and visualizes insights via a Streamlit dashboard.

---

## 📌 Project Overview

Online job platforms are increasingly exploited by scammers to post fake job listings. These listings can waste time, mislead job seekers, and even compromise personal information.

This project aims to build an **intelligent machine learning system** to detect fraudulent job postings based on their textual and categorical content. It includes:
- A binary classification model (`genuine` vs `fraudulent`)
- A real-time interactive dashboard built with **Streamlit**
- Support for uploading CSV job data and visualizing fraud insights

## ⚙️ Key Features & Technologies Used

### 🧠 Machine Learning
- **Model:** Logistic Regression (or Random Forest)
- **Features:** Text (title, description, etc.), categorical & boolean flags
- **Metrics:** F1-Score (important due to data imbalance)
- **Explainability:** SHAP values to interpret predictions

### API 
- We have created a API that returns JSON format output of Title, Prediction, Fraud Probaility.

### 🛠️ Tools & Libraries
- **Python**
- **Pandas, NumPy, Scikit-learn**
- **SHAP for explainability**
- **Streamlit** for dashboard UI
- **Matplotlib / Seaborn** for plotting

### 📊 Dashboard Includes:
- File upload interface
- Table of predictions with probabilities
- Histogram of fraud probabilities
- Pie chart: Real vs Fake jobs
- Top-10 most suspicious listings

  ###**Future Updates**
  - Functionality to view SHAP explanation
  - Display of SHAP summary Image
 
  ## 🚀 Setup Instructions (Step-by-Step)

> 🔁 Assumes Python 3.8+ installed

### 📁 1. Clone the Repository
### 📁 2. pip install -r requirements.txt
### 📁 3. streamlit run app.py 
  

## 📁 Project Structure
fraud_detection_job/<br>
│<br>
├── app.py # Streamlit dashboard<br>
├── model/<br>
│ ├── train_model.py # Model training script<br>
│ └── fraud_detector_pipeline.pkl # Saved model<br>
├── requirements.txt<br>
├── README.md<br>
└── data/<br>
├── train.csv<br>
└── test.csv<br>

**LINKS**
- **DEPLOYED LINK** - https://frauddetectionjobgit-3wnxugzqwsta8oqcpt7shd.streamlit.app/
- **VIDEO** - https://drive.google.com/file/d/14oX6PmoS95omNCKWJ7PzzvRRu7xCEMHB/view?usp=sharing

