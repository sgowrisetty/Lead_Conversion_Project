from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("final_random_forest_model.pkl")

@app.route("/")
def home():
    return "Lead Conversion Prediction API"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

# Commented for notebook submission
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)