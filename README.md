# NeuroScan AI 🧠⚡

> Multi-Model Ensemble Brain Tumor Classification & Visual Explainability System

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)

**NeuroScan AI** is a production-ready, full-stack medical computer vision application designed to assist in the early classification of brain tumors from MRI scans. It leverages a multi-model ensemble architecture (**DenseNet121**, **MobileNetV2**, and **VGG16**) alongside **Grad-CAM (Gradient-weighted Class Activation Mapping)** explainability heatmaps to localize tumor regions.

---<img width="1065" height="2549" alt="127 0 0 1-predict" src="https://github.com/user-attachments/assets/9279ee9c-352e-470d-94e3-2f955d29cf24" />


## 🚀 Key Features

* **Multi-Model Ensemble Diagnostics:** Runs parallel classification across DenseNet121, MobileNetV2, and VGG16 to generate robust cross-model confidence scores.
* **AI Grad-CAM Visual Heatmaps:** Highlights critical areas of the brain scan that heavily influenced the model's prediction, providing visual explainability.
* **DICOM & Standard Image Support:** Automatically parses DICOM medical files (`.dcm`) alongside standard PNG/JPG MRI formats.
* **Professional PDF Report Generation:** Exports an official summary report complete with patient metadata, ensemble predictions, medical images, and regional specialist hospital references.
* **Robust Keras Compatibility Layer:** Incorporates custom monkey-patches and sanitization routines to safely load legacy `.h5` model files under modern Keras/TensorFlow environments.

---

## 🛠️ Tech Stack

* **Backend & API:** Python, Flask, Flask-CORS, Werkzeug
* **Deep Learning & Computer Vision:** TensorFlow, Keras, OpenCV, NumPy, Pillow, PyDicom
* **Reporting:** ReportLab (Dynamic PDF generation)
* **Frontend:** HTML5, CSS3, JavaScript, Responsive UI Components
* **Hosting & Deployment:** Render

---

## 📂 Project Structure

```text
neuroscan-ai/
│
├── app.py                  # Main Flask application & routing logic
├── preprocess.py           # Data generators and class index loaders
├── models/
│   └── densenet_model.h5   # Pre-trained deep learning classification weights
├── static/
│   └── uploads/            # Temporary storage for uploaded scans and generated heatmaps
├── templates/
│   └── index.html          # Frontend user dashboard UI
├── hospitals_db.json       # Global hospital referral directory matching continental schemas
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation<img width="1065" height="2549" alt="127 0 0 1-predict" src="https://github.com/user-attachments/assets/63c7e50a-b3d4-4d26-907d-78650f33b9dd" />

## 🌐 Live Application
You can try out the live version of NeuroScan AI here:
👉  [**View Live Demo on Render**](https://neuroscan-ai-mk80.onrender.com)
