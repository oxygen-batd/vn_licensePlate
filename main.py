import os
from flask import Flask, request, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename
import torch
from ultralytics import YOLO
import cv2
from util import read_license_plate, limit_text_within_image
import numpy as np

app = Flask(__name__)

# Load YOLO model
model_path = 'models/best_model.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
license_plates = YOLO(model_path).to(device)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Yolov8 + PaddleOCR
def process_image(file_path, filename):
    image = cv2.imread(file_path)
    results = license_plates.predict(source=file_path, device=device)[0]

    for license_plate in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        license_plate_crop = image[y1:y2, x1:x2]

        license_plate_text = read_license_plate(license_plate_crop)

        if license_plate_text is not None:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Define text parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            
            # Limit text within image boundaries
            text_org = (x1, y1 - 10)
            text_org = limit_text_within_image(image, license_plate_text, text_org, font, font_scale, thickness)
            
            # Draw text on image
            cv2.putText(image, license_plate_text, text_org, font, font_scale, (36, 255, 12), thickness)
            
    output_image_path = f'static/output/{filename}'
    cv2.imwrite(output_image_path, image)

    return os.path.basename(output_image_path)

# Route for uploading files
@app.route('/')
def upload_form():
    return render_template('index.html')

# Route for handling file upload
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        output_filename = process_image(file_path, filename)
        return render_template('index.html', filename=output_filename)

    return redirect(request.url)

# uploaded
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# output
@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
