import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import timm
from torch import no_grad, inference_mode
from PatientFriend.ChatPatient import ChatPatient
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from flask import send_from_directory


UPLOAD_FOLDER = './upload_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
GENERATION_FOLDER = './generated'
os.makedirs(GENERATION_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("resnet18").to(device)
model.eval()



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

@no_grad()
@inference_mode()
def get_prediction(image_bytes, filepath, filename):
    rgb_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # Normalize the pixel values
    rgb_img = np.float32(rgb_img) / 255

    # Resize the image to 224x224
    rgb_img_resized = cv2.resize(rgb_img, (224, 224))

    tensor = transform_image(image_bytes=image_bytes).to(device)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    outputs = model.forward(tensor).cpu()
    _, y_hat = outputs.max(1)
    predicted_idx = int(y_hat.item())
    targets = [ClassifierOutputTarget(predicted_idx)]

    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)
    visualization = Image.fromarray(visualization)
    visualization.save(f"{GENERATION_FOLDER}/{filename}")
    return predicted_idx



@app.route(f'/{GENERATION_FOLDER}/<path:path>')
def send_report(path):
    return send_from_directory(f'{GENERATION_FOLDER}', path)

@app.route('/api/predict_tb', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_bytes = file.read()
            class_id = get_prediction(img_bytes, filepath, filename)
            return jsonify({'class_id': class_id})
    # For referencing this flask server in the frontend, use a action="http://..../api/predict_tb" in the form
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
chat_patient = ChatPatient()

@app.route('/api/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    report_text = data.get('text')

    #Replace the double quotes
    report_text = report_text.replace('"','')

    if not report_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        summary = chat_patient.get_friendly_text(report_text)
        return jsonify({'summary': summary}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get('text')

    #Replace the double quotes
    text = text.replace('"','')

    language = data.get('language', 'hindi')  # Default language is Hindi
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        translated_text = chat_patient.translate_text(text, language=language)
        return jsonify({'translated_text': translated_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
