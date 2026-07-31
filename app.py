# ============================================================
# app.py
# AI-ML Assignment 10 : Flask REST API for Heart Disease Prediction
# ============================================================

import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model bundle (model + scaler + feature order) at startup
bundle = joblib.load("model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Heart Disease Prediction API is running.",
        "usage": "POST patient data as JSON to /predict",
        "required_fields": feature_names
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Build the feature vector in the exact order used during training
        missing = [f for f in feature_names if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing required fields: {missing}"
            }), 400

        features = [float(data[f]) for f in feature_names]
        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "confidence": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # For local testing. Render will use gunicorn instead (see Procfile).
    app.run(host="0.0.0.0", port=5000, debug=True)
