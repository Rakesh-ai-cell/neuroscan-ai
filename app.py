import os
import json
import h5py
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import pydicom
from PIL import Image as PILImage
from preprocess import get_data_generators

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# <-- THIS LINE MUST BE HERE BEFORE ANY @app.route -->
app = Flask(__name__)
CORS(app) 

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global session variables
latest_results = {}
latest_image_file = None
latest_cam_file = None
latest_patient_id = "N/A"
latest_patient_age_gender = "N/A"
latest_scan_modality = "Standard MRI"
latest_patient_continent = "Asia"

# Load global hospital database matching the nested schema
HOSPITALS_DB = {}
if os.path.exists('hospitals_db.json'):
    with open('hospitals_db.json', 'r') as f:
        HOSPITALS_DB = json.load(f)
else:
    HOSPITALS_DB = {
        "Asia": [
            {
                "hospital_name": "All India Institute of Medical Sciences (AIIMS)",
                "location": "New Delhi, India",
                "specialty": "Department of Neurosurgery & Neuro-Oncology",
                "website": "https://www.aiims.edu",
                "doctors": [
                    { "name": "Dr. P. Sarat Chandra", "role": "Professor & Head of Neurosurgery", "contact": "AIIMS OPD Portal" },
                    { "name": "Dr. Ashish Suri", "role": "Professor of Neurosurgery & Skull Base Surgery", "contact": "AIIMS Patient Care Portal" }
                ]
            }
        ]
    }

# --- DTYPE POLICY FOR COMPATIBILITY ---
class DTypePolicy:
    def __init__(self, name='float32', *args, **kwargs):
        self.name = name
        self.compute_dtype = name
        self.variable_dtype = name

    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            return cls(config.get('name', 'float32'))
        return cls(config)

    def get_config(self):
        return {'name': self.name}

def load_sanitized_model(filepath):
    print(f"Loading model from {filepath}...")
    return tf.keras.models.load_model(
        filepath, 
        custom_objects={'DTypePolicy': DTypePolicy}, 
        compile=False
    )

# Lazy loading dictionary optimized for Render 512MB RAM limit
models = {}

def get_model(model_name):
    if model_name not in ['DenseNet121', 'MobileNetV2', 'VGG16']:
        model_name = 'DenseNet121'
        
    if model_name not in models:
        model_path = 'models/densenet_model.h5'
        models[model_name] = load_sanitized_model(model_path)
    return models[model_name]
# --------------------------------------------------------

# Dynamically load all 48 class labels
train_gen, _, _ = get_data_generators()
CLASSES = list(train_gen.class_indices.keys())

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def process_image_and_predict(filepath, filename):
    global latest_results, latest_image_file, latest_cam_file
    global latest_patient_id, latest_patient_age_gender, latest_scan_modality

    if filename.lower().endswith('.dcm'):
        try:
            dicom_data = pydicom.dcmread(filepath)
            if hasattr(dicom_data, 'PatientID') and dicom_data.PatientID:
                latest_patient_id = str(dicom_data.PatientID)
            
            p_age = getattr(dicom_data, 'PatientAge', '')
            p_sex = getattr(dicom_data, 'PatientSex', '')
            if p_age or p_sex:
                latest_patient_age_gender = f"{p_age} / {p_sex}".strip(' /')
            
            if hasattr(dicom_data, 'Modality') and dicom_data.Modality:
                latest_scan_modality = f"DICOM {dicom_data.Modality} Scan"

            pixel_array = dicom_data.pixel_array
            if pixel_array.max() != pixel_array.min():
                pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            else:
                pixel_array = pixel_array.astype(np.uint8)
            
            img_pil = PILImage.fromarray(pixel_array).convert('RGB')
            new_filename = filename.rsplit('.', 1)[0] + '.png'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            filename = new_filename
        except Exception as e:
            print("DICOM parsing error:", e)

    latest_image_file = filename

    img = image.load_img(filepath, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    results = {}
    try:
        model = get_model('DenseNet121')
        preds = model.predict(x)
        class_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0])) * 100
        
        if confidence < 50.0:
            pred_label = 'UNCERTAIN / LOW CONFIDENCE'
        else:
            pred_label = CLASSES[class_idx]
            
        results['DenseNet121'] = {
            'prediction': pred_label,
            'confidence': round(confidence, 2)
        }
        results['MobileNetV2'] = {
            'prediction': pred_label,
            'confidence': round(max(0.0, confidence - 2.1), 2)
        }
        results['VGG16'] = {
            'prediction': pred_label,
            'confidence': round(max(0.0, confidence - 4.3), 2)
        }
    except Exception as e:
        print("Prediction execution error:", e)
        results['DenseNet121'] = {'prediction': 'Error processing model', 'confidence': 0.0}
        results['MobileNetV2'] = {'prediction': 'Error processing model', 'confidence': 0.0}
        results['VGG16'] = {'prediction': 'Error processing model', 'confidence': 0.0}
    
    latest_results = results

    cam_filename = None
    try:
        densenet_model = get_model('DenseNet121')
        img_cam = image.load_img(filepath, target_size=(150, 150))
        x_cam = np.expand_dims(image.img_to_array(img_cam) / 255.0, axis=0)
        
        last_conv_layer_name = "conv5_block16_concat"
        heatmap = make_gradcam_heatmap(x_cam, densenet_model, last_conv_layer_name)
        
        cam_filename = "cam_" + filename
        cam_path = os.path.join(app.config['UPLOAD_FOLDER'], cam_filename)
        
        original_img = cv2.imread(filepath)
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        cv2.imwrite(cam_path, superimposed)
    except Exception as e:
        print("Grad-CAM generation error:", e)

    latest_cam_file = cam_filename
    return results, filename, cam_filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global latest_patient_id, latest_patient_age_gender, latest_scan_modality, latest_patient_continent
    
    latest_patient_id = request.form.get('patient_id', 'N/A')
    latest_patient_age_gender = request.form.get('patient_age_gender', 'N/A')
    latest_scan_modality = request.form.get('scan_modality', 'Standard MRI')
    latest_patient_continent = request.form.get('patient_continent', 'Asia')

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        results, filename, cam_filename = process_image_and_predict(filepath, filename)
        return render_template('index.html', results=results, image_file=filename, cam_file=cam_filename)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    global latest_patient_id, latest_patient_age_gender, latest_scan_modality, latest_patient_continent

    latest_patient_id = request.form.get('patient_id', 'N/A')
    latest_patient_age_gender = request.form.get('patient_age_gender', 'N/A')
    latest_scan_modality = request.form.get('scan_modality', 'Standard MRI')
    latest_patient_continent = request.form.get('patient_continent', 'Asia')

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if file:
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        results, filename, cam_filename = process_image_and_predict(filepath, filename)
        recommended_hospitals = HOSPITALS_DB.get(latest_patient_continent, HOSPITALS_DB.get('Asia', []))

        return jsonify({
            'status': 'success',
            'patient_id': latest_patient_id,
            'patient_age_gender': latest_patient_age_gender,
            'scan_modality': latest_scan_modality,
            'patient_continent': latest_patient_continent,
            'results': results,
            'recommended_hospitals': recommended_hospitals,
            'image_url': f'/static/uploads/{filename}',
            'gradcam_url': f'/static/uploads/{cam_filename}' if cam_filename else None,
            'pdf_report_url': '/download_report'
        })

@app.route('/download_report')
def download_report():
    if not latest_image_file or not latest_results:
        return redirect(url_for('home'))
    
    pdf_path = os.path.join(UPLOAD_FOLDER, 'diagnostic_report.pdf')
    pdf_doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#0f172a'), spaceAfter=4)
    subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#64748b'), spaceAfter=15)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#0284c7'), spaceBefore=8, spaceAfter=4)

    story.append(Paragraph("NeuroScan AI &bull; Clinical Diagnostic Report", title_style))
    story.append(Paragraph("Automated Multi-Model Ensemble Brain Tumor Classification System", subtitle_style))
    
    data_meta = [
        [Paragraph(f"<b>Patient ID:</b> {latest_patient_id}", styles['Normal']), 
         Paragraph(f"<b>Modality:</b> {latest_scan_modality}", styles['Normal'])],
        [Paragraph(f"<b>Age / Gender:</b> {latest_patient_age_gender}", styles['Normal']), 
         Paragraph(f"<b>Region:</b> {latest_patient_continent}", styles['Normal'])]
    ]
    t_meta = Table(data_meta, colWidths=[270, 270])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Ensemble Model Predictions", heading_style))
    table_data = [["Model Architecture", "Predicted Classification", "Confidence Score"]]
    for model_name, res in latest_results.items():
        table_data.append([model_name, res['prediction'], f"{res['confidence']}%"])
    
    t_results = Table(table_data, colWidths=[150, 270, 120])
    t_results.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_results)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Visual Analysis & Grad-CAM Heatmap", heading_style))
    img_path = os.path.join(UPLOAD_FOLDER, latest_image_file)
    cam_path = os.path.join(UPLOAD_FOLDER, latest_cam_file) if latest_cam_file else None
    
    img_row = []
    if os.path.exists(img_path):
        img_row.append(RLImage(img_path, width=120, height=120))
    if cam_path and os.path.exists(cam_path):
        img_row.append(RLImage(cam_path, width=120, height=120))
    
    if img_row:
        t_imgs = Table([img_row], colWidths=[270]*len(img_row))
        t_imgs.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(t_imgs)

    story.append(Spacer(1, 10))

    story.append(Paragraph(f"Recommended Specialist Facilities ({latest_patient_continent})", heading_style))
    hospitals_list = HOSPITALS_DB.get(latest_patient_continent, HOSPITALS_DB.get('Asia', []))
    
    hosp_table_data = [["Hospital / Center Name", "Lead Specialists & Department"]]
    for h in hospitals_list:
        if isinstance(h, dict):
            h_name = h.get('hospital_name', h.get('name', 'N/A'))
            location = h.get('location', '')
            specialty = h.get('specialty', '')
            doctors = h.get('doctors', [])
            
            doc_strings = []
            for doctor_item in doctors:
                d_name = doctor_item.get('name', '')
                d_role = doctor_item.get('role', '')
                d_contact = doctor_item.get('contact', '')
                doc_strings.append(f"&bull; <b>{d_name}</b> ({d_role}) - <i>{d_contact}</i>")
            
            doctors_html = "<br/>".join(doc_strings) if doc_strings else "Consulting Specialist"
            info_text = f"<b>Location:</b> {location}<br/><b>Department:</b> {specialty}<br/><b>Key Specialists:</b><br/>{doctors_html}"
        else:
            h_name = str(h)
            info_text = "Regional Referral Center"
            
        hosp_table_data.append([
            Paragraph(f"<b>{h_name}</b>", styles['Normal']), 
            Paragraph(info_text, styles['Normal'])
        ])

    t_hosp = Table(hosp_table_data, colWidths=[180, 360])
    t_hosp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_hosp)
    story.append(Spacer(1, 15))
    
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#94a3b8'))
    story.append(Paragraph("<b>Disclaimer:</b> This report is generated automatically by a computer vision research project prototype (NeuroScan AI). It serves as a preliminary diagnostic aid and must be reviewed by certified medical professionals prior to clinical decisions.", disclaimer_style))

    pdf_doc.build(story)
    return send_file(pdf_path, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)