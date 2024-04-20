import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import timm
from torch import no_grad, inference_mode


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


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id})
    # For referencing this flask server in the frontend, use a action="http://..../predict" in the form
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    app.run(debug=True)
