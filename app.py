"""Flask API. ESP32 / mobile app POSTs sensor data here and gets the pain-risk back.
Run: python src/app.py     then POST to http://<pc-ip>:5000/predict"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from flask import Flask, request, jsonify
from predictor import PainPredictor
from alert import send_alert

app = Flask(__name__)
predictor = PainPredictor()


@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json(force=True)
    res = predictor.predict(bone=d.get("bone"), emg=d.get("emg"), skin=d.get("skin"))
    if res["level"] == "HIGH":
        send_alert(res, user=d.get("user", "patient-1"))
        res["alert_sent"] = True
    return jsonify(res)


@app.route("/")
def home():
    return "Pain risk API running. POST JSON to /predict"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
