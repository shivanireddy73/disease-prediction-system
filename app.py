from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Load trained model
model = joblib.load("disease_model.pkl")

# Home route
@app.route('/')
def home():
    return "Disease Prediction API Running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    features = np.array(data['features']).reshape(1, -1)

    prediction = model.predict(features)

    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

    return jsonify({
        "prediction": result
    })

# Run server
if __name__ == '__main__':
    app.run(debug=True) 