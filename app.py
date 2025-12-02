from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

# Load model
with open("model/model.json") as f:
    model = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        population = float(data["population"])
        w = model["theta1"]
        b = model["theta0"]
        prediction = w * population + b
        return jsonify({"predicted_profit": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)