# 🏥 Multi-Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.2+-black.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end ML system to predict **Diabetes**, **Heart Disease**, and **Breast Cancer** with a Streamlit web app and Flask REST API.

---

## 📊 Model Results

| Disease | Best Model | AUC-ROC | F1 Score | Accuracy |
|---|---|---|---|---|
| **Diabetes** | Logistic Regression | 0.9087 | 0.7965 | 0.8506 |
| **Heart Disease** | Logistic Regression | 0.9102 | 0.8406 | 0.8197 |
| **Breast Cancer** | Logistic Regression | 1.0000 | 1.0000 | 1.0000 |

---

## 🚀 Getting Started

```bash
git clone https://github.com/dimpalshegekar/disease-prediction.git
cd disease-prediction
pip install -r requirements.txt
python train.py
streamlit run app.py      # Web app → http://localhost:8501
python api.py             # Flask API → http://localhost:5000
```

---

## 🔗 Flask API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | API health check |
| `/predict/diabetes` | POST | Predict diabetes risk |
| `/predict/heart` | POST | Predict heart disease risk |
| `/predict/cancer` | POST | Predict breast cancer risk |

### Example API Call
```bash
curl -X POST http://localhost:5000/predict/diabetes \
  -H "Content-Type: application/json" \
  -d '{"Glucose": 148, "BMI": 33.6, "Age": 50, "Pregnancies": 6}'
```

---

## 👩‍💻 Author
**Dimpal Shegekar** — [@dimpalshegekar](https://github.com/dimpalshegekar)

## 📄 License
MIT License
