from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

app = Flask(__name__)

# Feature names in the exact order used during training
FEATURE_NAMES = [
    "baseline value", "accelerations", "fetal_movement", "uterine_contractions",
    "light_decelerations", "severe_decelerations", "prolongued_decelerations",
    "abnormal_short_term_variability", "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability", "histogram_width", "histogram_min",
    "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes",
    "histogram_mode", "histogram_mean", "histogram_median",
    "histogram_variance", "histogram_tendency"
]

# Load model using native XGBoost format (no version warnings)
model = XGBClassifier()
model.load_model('fetalai_model.json')

# Load scaler
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle both form data (from browser) and JSON (from API requests)
    if request.is_json:
        data = request.get_json(force=True)
    else:
        data = request.form.to_dict()
        data = {key: float(value) for key, value in data.items()}

    # FIX: Use a DataFrame with proper feature names to avoid sklearn warning
    features_df = pd.DataFrame([data], columns=FEATURE_NAMES)
    features_scaled = scaler.transform(features_df)

    prediction = model.predict(features_scaled)[0]
    # Adjust prediction back to 1, 2, 3 for user
    prediction = int(prediction) + 1

    # Map prediction to a label
    labels = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    result = labels[prediction]
    message = (
        "Normal fetal health."
        if prediction == 1
        else "Potential concern detected. Consult a healthcare provider."
    )

    if request.is_json:
        return jsonify({'fetal_health': prediction})
    else:
        return render_template('result.html', result=result, prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
