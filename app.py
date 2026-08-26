import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
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

app = Flask(__name__)

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

# Load top reliable models into a dictionary for comparison
models = {
    'DenseNet121': load_model('models/densenet_model.h5'),
    'MobileNetV2': load_model('models/mobilenet_model.h5'),
    'VGG16': load_model('models/vgg16_model.h5')
}

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global latest_results, latest_image_file, latest_cam_file
    global latest_patient_id, latest_patient_age_gender, latest_scan_modality
    
    # Capture fallback manual form metadata
    latest_patient_id = request.form.get('patient_id', 'N/A')
    latest_patient_age_gender = request.form.get('patient_age_gender', 'N/A')
    latest_scan_modality = request.form.get('scan_modality', 'Standard MRI')

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

        # Advanced DICOM Handling (.dcm files)
        if filename.lower().endswith('.dcm'):
            try:
                dicom_data = pydicom.dcmread(filepath)
                
                # Automatically extract hidden metadata if available
                if hasattr(dicom_data, 'PatientID') and dicom_data.PatientID:
                    latest_patient_id = str(dicom_data.PatientID)
                
                p_age = getattr(dicom_data, 'PatientAge', '')
                p_sex = getattr(dicom_data, 'PatientSex', '')
                if p_age or p_sex:
                    latest_patient_age_gender = f"{p_age} / {p_sex}".strip(' /')
                
                if hasattr(dicom_data, 'Modality') and dicom_data.Modality:
                    latest_scan_modality = f"DICOM {dicom_data.Modality} Scan"

                # Convert DICOM pixel array to viewable PNG image
                pixel_array = dicom_data.pixel_array
                if pixel_array.max() != pixel_array.min():
                    pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
                else:
                    pixel_array = pixel_array.astype(np.uint8)
                
                img_pil = PILImage.fromarray(pixel_array).convert('RGB')
                new_filename = filename.rsplit('.', 1)[0] + '.png'
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                img_pil.save(filepath)
                filename = new_filename
            except Exception as e:
                print("DICOM parsing error:", e)

        latest_image_file = filename

        # Preprocess image for model prediction
        img = image.load_img(filepath, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Run ensemble predictions with confidence thresholding
        results = {}
        for name, model in models.items():
            preds = model.predict(x)
            class_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0])) * 100
            
            if confidence < 50.0:
                results[name] = {
                    'prediction': 'UNCERTAIN / LOW CONFIDENCE',
                    'confidence': round(confidence, 2)
                }
            else:
                results[name] = {
                    'prediction': CLASSES[class_idx],
                    'confidence': round(confidence, 2)
                }
        
        latest_results = results

        # Generate Grad-CAM Heatmap using DenseNet121
        cam_filename = None
        try:
            densenet_model = models['DenseNet121']
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
        return render_template('index.html', results=results, image_file=filename, cam_file=cam_filename)

@app.route('/download_report')
def download_report():
    if not latest_image_file or not latest_results:
        return redirect(url_for('home'))
    
    pdf_path = os.path.join(UPLOAD_FOLDER, 'diagnostic_report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#0f172a'), spaceAfter=4)
    subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#64748b'), spaceAfter=15)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#0284c7'), spaceBefore=8, spaceAfter=4)

    story.append(Paragraph("NeuroScan AI &bull; Clinical Diagnostic Report", title_style))
    story.append(Paragraph("Automated Multi-Model Ensemble Brain Tumor Classification System", subtitle_style))
    
    # Metadata Table reflecting auto-extracted DICOM values or form values
    data_meta = [
        [Paragraph(f"<b>Patient ID:</b> {latest_patient_id}", styles['Normal']), 
         Paragraph(f"<b>Modality:</b> {latest_scan_modality}", styles['Normal'])],
        [Paragraph(f"<b>Age / Gender:</b> {latest_patient_age_gender}", styles['Normal']), 
         Paragraph("<b>Status:</b> Verified Analysis", styles['Normal'])]
    ]
    t_meta = Table(data_meta, colWidths=[270, 270])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Results Table
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

    # Images Section
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

    story.append(Spacer(1, 15))
    
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#94a3b8'))
    story.append(Paragraph("<b>Disclaimer:</b> This report is generated automatically by a computer vision research project prototype (NeuroScan AI). It serves as a preliminary diagnostic aid and must be reviewed by certified medical professionals prior to clinical decisions.", disclaimer_style))

    doc.build(story)
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)