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
# TODO: Grad cam is missing here and also do any possible optimizations

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("resnet18").to(device)
model.eval()


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
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor.to(device)).cpu()
    _, y_hat = outputs.max(1)
    predicted_idx = int(y_hat.item())
    return predicted_idx


@app.route('/api/predict_tb', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id = get_prediction(image_bytes=img_bytes)
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
