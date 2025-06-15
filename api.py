from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model/fraud_detector_pipeline.pkl")

@app.route("/")
def home():
    return "✅ Fraud Detection API is running!"

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)

    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    num_cols = ['telecommuting', 'has_company_logo', 'has_questions']
    df['text'] = df[text_cols].fillna("").agg(" ".join, axis=1)
    X_input = df[['text'] + cat_cols + num_cols]

    probabilities = model.predict_proba(X_input)[:, 1]
    predictions = model.predict(X_input)

    result = []
    for i in range(len(df)):
        result.append({
            "title": df.iloc[i]['title'],
            "prediction": int(predictions[i]),
            "fraud_probability": float(probabilities[i])
        })

    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000, debug=True)