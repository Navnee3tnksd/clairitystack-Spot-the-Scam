import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

#load model
model = joblib.load("model/fraud_detector_pipeline.pkl")
sample_data = pd.read_csv('data/0tkf3jUGLYjCEJGz.csv').sample(100)

text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
num_cols = ['telecommuting', 'has_company_logo', 'has_questions']
sample_data['text'] = sample_data[text_cols].fillna('').agg(" ".join, axis=1)
X_sample = sample_data[['text'] + cat_cols + num_cols]

X_transformed = model.named_steps['preprocess'].transform(X_sample)
explainer = shap.Explainer(model.named_steps['clf'], X_transformed)
shap_values = explainer(X_transformed)


#Save Shap plot
shap.summary_plot(shap_values, X_transformed, show = False)

plt.savefig("shap_summary.png")