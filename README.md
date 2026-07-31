# Heart Disease Prediction — End-to-End ML Deployment

## Objective
Build a machine learning model that predicts whether a patient is at risk
of heart disease based on clinical parameters, expose it as a REST API
using Flask, and deploy it as a live web service on Render.

## Dataset
**Heart Disease Prediction Dataset** (Kaggle)
Link: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

## Libraries Used
- pandas — data loading and manipulation
- numpy — numerical operations
- scikit-learn — train/test split, feature scaling, Logistic Regression model, evaluation
- joblib — model serialization
- Flask — REST API framework
- gunicorn — production WSGI server (used by Render)

## Methodology
1. **Data Understanding & Preprocessing**
   - Loaded `heart.csv` with pandas and inspected the first five records.
   - Identified all clinical measurements (age, cholesterol, blood pressure, etc.) as numerical features, with `target` as the binary label (1 = heart disease present, 0 = absent).
   - Checked for missing values and handled any found.
   - Split the data 80% training / 20% testing using a stratified split to preserve class balance.
   - Standardized features with `StandardScaler` so all inputs are on a comparable scale.

2. **Model Development**
   - Trained a **Logistic Regression** classifier on the scaled training data.
   - Evaluated on the held-out test set using accuracy score.
   - Bundled the trained model, the fitted scaler, and the exact feature order into a single `model.pkl` file using `joblib`, so the API can reproduce identical preprocessing at inference time.

3. **API Development**
   - Built a Flask app (`app.py`) that loads `model.pkl` on startup.
   - `GET /` returns basic usage info and the list of required input fields.
   - `POST /predict` accepts patient details as JSON, validates that all required fields are present, scales the input the same way as training, and returns a JSON prediction plus a confidence score.

4. **Deployment**
   - Pushed the full project to a public GitHub repository.
   - Deployed the Flask app on **Render** as a web service, using `gunicorn` (declared in the `Procfile`) as the production server.

## CNN / Model Architecture
Not applicable here — this assignment uses a tabular classifier
(Logistic Regression), not a CNN. The pipeline is:

```
Raw patient data (JSON)
        │
        ▼
Feature ordering + StandardScaler
        │
        ▼
Logistic Regression classifier
        │
        ▼
Prediction: "Heart Disease Detected" / "No Heart Disease Detected"
        + confidence score
```

## Results
| Metric | Value |
|---|---|
| Test Accuracy | 0.8098 |



## API Usage Example

**Request**
```bash
curl -X POST https://<your-render-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
        "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
        "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
      }'
```

**Response**
```json
{
  "prediction": "Heart Disease Detected",
  "confidence": 0.83
}
```

## Live Deployment
Render URL: **[Heart_disease_prediction_link](https://assignment-10-8vqh.onrender.com)**

## Conclusion
This project trained a Logistic Regression model to predict heart disease
risk from clinical parameters, achieving solid test accuracy on the held-out
split (see Results above). The model, scaler, and feature order were
serialized together to guarantee that predictions served through the API
match exactly what was learned during training. The main challenge during
deployment was ensuring the Flask app's preprocessing pipeline stayed
perfectly consistent with training-time preprocessing, and configuring
Render to use gunicorn instead of Flask's development server for a stable,
publicly accessible endpoint. This project highlights why MLOps practices
matter: a model is only useful once it can be reliably packaged, versioned,
served, and monitored in production, rather than living only inside a
notebook. Proper serialization, API design, and cloud deployment turn a
one-off experiment into a service other systems and users can actually
depend on.

## Repository Structure
```
HeartDiseaseDeployment/
├── app.py
├── model.pkl
├── requirements.txt
├── Procfile
├── train_model.py
├── heart.csv
└── README.md
```
