from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import json
import requests
import threading
import uuid
import os

app = Flask(__name__)

BASE = os.path.join(os.path.dirname(__file__), "models")

print("Loading model... please wait")
model     = tf.keras.models.load_model(
    os.path.join(BASE, "ecg_model.keras"), compile=False)
scaler    = joblib.load(os.path.join(BASE, "scaler.save"))
threshold = joblib.load(os.path.join(BASE, "threshold.save"))

with open(os.path.join(BASE, "feature_indices.json")) as f:
    feat_meta = json.load(f)

TOP_INDICES    = feat_meta["top_15_indices"]
FEATURE_LABELS = feat_meta["feature_labels"]
IMPORTANCES    = feat_meta["top_15_importances"]
N_FEATURES     = feat_meta["n_features"]

FOG_URL = "http://127.0.0.1:5001"

print(f"Model ready — {N_FEATURES} features: {FEATURE_LABELS}")
print(f"Threshold: {threshold:.6f}")
print("Open http://127.0.0.1:5000\n")


def load_csv():
    """
    Searches multiple locations for the MIT-BIH CSV file.
    Checks test set first, falls back to train set.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    home_dir   = os.path.expanduser("~")

    possible_paths = [
        # Same folder as app.py
        os.path.join(script_dir, "mitbih_test.csv"),
        os.path.join(script_dir, "mitbih_train.csv"),
        # data/ subfolder
        os.path.join(script_dir, "data", "mitbih_test.csv"),
        os.path.join(script_dir, "data", "mitbih_train.csv"),
        # Downloads folder (Mac)
        os.path.join(home_dir, "Downloads", "mitbih_test.csv"),
        os.path.join(home_dir, "Downloads", "mitbih_train.csv"),
        # Desktop (Mac)
        os.path.join(home_dir, "Desktop", "mitbih_test.csv"),
        os.path.join(home_dir, "Desktop", "mitbih_train.csv"),
        # Current working directory
        "mitbih_test.csv",
        "mitbih_train.csv",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"CSV found: {path}")
            df = pd.read_csv(path, header=None)
            return df.iloc[:, :-1].values, df.iloc[:, -1].values

    print("ERROR: mitbih_test.csv not found in any location.")
    print("Please copy it to your project folder:", script_dir)
    return None, None


def push_to_fog(session_id, beat_index, is_anomaly, loss_value):
    try:
        requests.post(f"{FOG_URL}/fog/ingest", json={
            "session_id": session_id,
            "beat_index": beat_index,
            "is_anomaly": bool(is_anomaly),
            "loss_value": float(loss_value)
        }, timeout=0.5)
    except Exception:
        pass


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/meta")
def meta():
    return jsonify({
        "feature_labels": FEATURE_LABELS,
        "importances":    IMPORTANCES,
        "n_features":     N_FEATURES,
        "threshold":      round(float(threshold), 6)
    })


@app.route("/sample/<sample_type>")
def sample(sample_type):
    try:
        X, y = load_csv()
        if X is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            return jsonify({
                "error": f"mitbih_test.csv not found. Please copy it to: {script_dir}"
            }), 404

        if sample_type == "normal":
            candidates = [i for i in range(len(y)) if y[i] == 0]
            if not candidates:
                return jsonify({"error": "No normal samples found in dataset"}), 404
            row_idx = candidates[0]

        elif sample_type == "abnormal":
            candidates = [i for i in range(len(y)) if y[i] != 0]
            if not candidates:
                return jsonify({"error": "No abnormal samples found in dataset"}), 404
            row_idx = candidates[0]

        else:
            row_idx = int(sample_type)
            if row_idx >= len(y):
                return jsonify({"error": f"Row {row_idx} out of range"}), 400

        top_vals = X[row_idx][TOP_INDICES].tolist()
        label    = int(y[row_idx])

        return jsonify({
            "values":       [round(float(v), 4) for v in top_vals],
            "row_index":    row_idx,
            "actual_label": label,
            "label_name":   "Normal" if label == 0 else f"Abnormal class {label}",
            "n_features":   N_FEATURES
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data       = request.json or {}
    features   = data.get("features", [])
    session_id = data.get("session_id", str(uuid.uuid4())[:8])
    beat_index = data.get("beat_index", 0)

    if len(features) != N_FEATURES:
        return jsonify({
            "error": f"Expected {N_FEATURES} values, got {len(features)}."
        }), 400

    try:
        X       = np.array(features, dtype=np.float32).reshape(1, N_FEATURES)
        X_sc    = scaler.transform(X)
        recon   = model.predict(X_sc, verbose=0)
        loss    = float(np.mean(np.square(X_sc - recon)))
        is_anom = bool(loss > threshold)
        conf    = round(
            min(abs(loss - float(threshold)) / float(threshold) * 100, 99.9), 1)

        t = threading.Thread(
            target=push_to_fog,
            args=(session_id, beat_index, is_anom, loss)
        )
        t.daemon = True
        t.start()

        return jsonify({
            "loss":       round(loss, 6),
            "threshold":  round(float(threshold), 6),
            "is_anomaly": is_anom,
            "confidence": conf,
            "session_id": session_id,
            "layer":      "edge"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fog/summary")
def fog_summary():
    try:
        r = requests.get(f"{FOG_URL}/fog/summary", timeout=2)
        return jsonify(r.json())
    except Exception:
        return jsonify({"error": "Fog node unreachable"}), 503


@app.route("/fog/clear", methods=["POST"])
def fog_clear():
    try:
        requests.post(f"{FOG_URL}/fog/clear", timeout=2)
        return jsonify({"status": "cleared"})
    except Exception:
        return jsonify({"error": "Fog node unreachable"}), 503


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)