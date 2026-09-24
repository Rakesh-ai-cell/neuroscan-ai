# NeuroScan AI 🧠⚡
> Multi-Model Ensemble Brain Tumor Classification & Visual Explainability System

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)

**NeuroScan AI** is a production-ready, full-stack medical computer vision application designed to assist in the early classification of brain tumors from MRI scans. It leverages a multi-model ensemble architecture (**DenseNet121**, **MobileNetV2**, and **VGG16**) alongside **Grad-CAM (Gradient-weighted Class Activation Mapping)** explainability heatmaps to localize tumor regions.

---

## 🌐 Live Application
You can try out the live version of NeuroScan AI here:  
👉 <a href="https://neuroscan-ai-mk80.onrender.com" target="_blank"><b>View Live Demo on Render</b></a>

---

## 🚀 Key Features

* **Multi-Model Ensemble Diagnostics:** Runs parallel classification across DenseNet121, MobileNetV2, and VGG16 to generate robust cross-model confidence scores.
* **AI Grad-CAM Visual Heatmaps:** Highlights critical areas of the brain scan that heavily influenced the model's prediction, providing visual explainability.
* **DICOM & Standard Image Support:** Automatically parses DICOM medical files (`.dcm`) alongside standard PNG/JPG MRI formats.
* **Professional PDF Report Export:** Compiles patient metadata, multi-model predictions, images, and regional specialist hospital references into an official clinical document.
* **Robust Keras Compatibility Layer:** Incorporates custom monkey-patches and sanitization routines to safely load legacy `.h5` model files under modern Keras/TensorFlow environments.

---

## 🖼️ Application Interface & Reports

### Web Dashboard Preview
<p align="center">
  <img src="https://raw.githubusercontent.com/Rakesh-ai-cell/neuroscan-ai/main/static/dashboard_preview.jpg" width="700" alt="NeuroScan AI Dashboard Preview">
</p>

### Clinical PDF Report Sample
<p align="center">
  <img src="https://raw.githubusercontent.com/Rakesh-ai-cell/neuroscan-ai/main/static/diagnostic_report.jpg" width="700" alt="Diagnostic Report Sample">
</p>

---

## 🛠️ Tech Stack

* **Backend & API:** Python, Flask, Flask-CORS, Werkzeug
* **Deep Learning & Computer Vision:** TensorFlow, Keras, OpenCV, NumPy, Pillow, PyDicom
* **Medical Reporting:** ReportLab (Dynamic PDF generation)
* **Frontend:** HTML5, CSS3, JavaScript, Responsive Glassmorphism UI Components
* **Hosting & Deployment:** Render Cloud Platform

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
└── README.md               # Project documentation
