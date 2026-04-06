# 🫀 ECG Edge Anomaly Detection System

## 📌 Overview

This project presents an **ECG Anomaly Detection System** using an Autoencoder-based deep learning model deployed in an **Edge Computing environment**.
It detects abnormal heartbeats from ECG signals in real-time with high reliability and low latency.

The system is designed to be **lightweight, efficient, and privacy-preserving**, making it suitable for healthcare edge devices.

---

## 🎯 Objectives

* Detect abnormal ECG signals (arrhythmia)
* Perform real-time inference on edge devices
* Reduce dependency on cloud processing
* Ensure patient data privacy
* Provide a user-friendly web interface

---

## ⚙️ Tech Stack

### 🧠 Machine Learning

* TensorFlow / Keras
* Autoencoder Neural Network
* Scikit-learn

### 💻 Backend

* Python
* Flask

### 🎨 Frontend

* HTML, CSS, JavaScript

### 📊 Data Processing

* Pandas
* NumPy
* Joblib

---

## 📂 Project Structure

```
ECG_Edge_Project/
│
├── app.py                  # Main Flask application
├── inference.py           # Model inference logic
├── fog_node.py            # Fog layer simulation
├── get_sample.py          # Sample ECG input generator
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
│
├── models/
│   ├── ecg_model.keras    # Trained Autoencoder model
│   ├── scaler.save        # Data scaler
│   └── threshold.save     # Threshold value
│
├── templates/
│   └── index.html         # Frontend UI
│
├── static/                # CSS / JS files
│
├── data/
│   └── mitbih_test.csv    # Sample dataset
```

---

## 🧠 Model Explanation

The system uses an **Autoencoder Neural Network**:

* Trained only on **normal ECG signals**
* Learns to reconstruct normal patterns
* Abnormal signals produce **high reconstruction error**

### 🔍 Detection Logic

```
If Reconstruction Error > Threshold → Abnormal
Else → Normal
```

---

## 📊 Performance Metrics

* Accuracy: ~90%
* Recall: High (important for healthcare)
* Precision: Balanced (avoids false alarms)
* AUC Score: ~0.90+

> The model is optimized to prioritize **recall**, ensuring abnormal cases are not missed.

---

## 🏗️ System Architecture

* **Edge Layer**: Runs model inference locally
* **Fog Layer**: Intermediate processing (optional)
* **Cloud Layer**: Model training & storage

---

## 🔄 Workflow

1. Input ECG signal
2. Preprocessing & normalization
3. Pass through Autoencoder
4. Compute reconstruction error
5. Compare with threshold
6. Output: Normal / Abnormal

---

## 🚀 How to Run

### 1️⃣ Clone Repository

```
git clone https://github.com/Arcanine19-prog/ECG-Edge-Anomaly-Detection.git
cd ECG-Edge-Anomaly-Detection
```

### 2️⃣ Create Virtual Environment (Mac)

```
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run Application

```
First run python fog_node.py then
python app.py
```

### 5️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 🌐 Features

✔ Real-time ECG anomaly detection
✔ Interactive web interface
✔ Edge computing support
✔ Lightweight model
✔ Fast inference

---

## 📈 Future Improvements

* CNN-based ECG model (higher accuracy)
* Real-time sensor integration
* Mobile app deployment
* Cloud dashboard for monitoring

---

## 🧑‍💻 Author

**Prasenjit Choudhury**
B.Tech Student, VIT Chennai

---

## 📜 License

This project is developed for academic purposes.

---

## ⭐ Acknowledgements

* MIT-BIH Arrhythmia Dataset
* TensorFlow & Open-source community
