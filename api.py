"""
Multi-Disease Prediction System - Flask API
Endpoints:
  POST /predict/diabetes
  POST /predict/heart
  POST /predict/cancer
  GET  /health
"""
from flask import Flask, request, jsonify
import joblib, os, numpy as np, pandas as pd

app = Flask(__name__)

def load_model(tag):
    base = os.path.dirname(os.path.abspath(__file__))
    for folder in [os.path.join(base,"models"), os.path.join(os.getcwd(),"models")]:
        mp = os.path.join(folder, f"{tag}_model.pkl")
        if os.path.exists(mp):
            return (joblib.load(mp),
                    joblib.load(os.path.join(folder, f"{tag}_scaler.pkl")),
                    joblib.load(os.path.join(folder, f"{tag}_features.pkl")))
    return None, None, None

def predict(tag, data):
    model, scaler, features = load_model(tag)
    if model is None:
        return {"error": "Model not found. Run train.py first."}, 500
    try:
        vals = [data.get(f, 0) for f in features]
        X = pd.DataFrame([vals], columns=features)
        X_scaled = scaler.transform(X)
        prob = float(model.predict_proba(X_scaled)[0][1])
        pred = int(model.predict(X_scaled)[0])
        risk = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
        return {"prediction": pred, "probability": round(prob, 4),
                "risk_level": risk, "status": "success"}, 200
    except Exception as e:
        return {"error": str(e)}, 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "Disease Prediction API"})

@app.route("/predict/diabetes", methods=["POST"])
def diabetes():
    res, code = predict("diabetes", request.json or {})
    return jsonify(res), code

@app.route("/predict/heart", methods=["POST"])
def heart():
    res, code = predict("heart_disease", request.json or {})
    return jsonify(res), code

@app.route("/predict/cancer", methods=["POST"])
def cancer():
    res, code = predict("breast_cancer", request.json or {})
    return jsonify(res), code

if __name__ == "__main__":
    print("🏥 Disease Prediction API running on http://localhost:5000")
    app.run(debug=True, port=5000)
