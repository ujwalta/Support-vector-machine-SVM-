from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model and scaler
with open("model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "SVM Credit Card Eligibility API. Use /predict endpoint with JSON data."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Extract features from JSON
        income = float(data["income"])
        credit_score = float(data["credit_score"])
        debt = float(data["debt"])

        # Prepare input data
        input_data = np.array([[income, credit_score, debt]])
        input_data = scaler.transform(input_data)  # Scale input

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return result
        result = "Eligible" if prediction == 1 else "Not Eligible"
        return jsonify({"Credit_Card_Eligibility": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
