import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model = load_model('BrainTumorDetection.h5')


def get_labels(label):
	if label==0:
		return "No"
	elif label==1:
		return "Yes"


def predict(img_path):
    img = cv2.imread(img_path)
    img_array = Image.fromarray(img, 'RGB')
    resized_img = img_array.resize((64, 64))
    resized_img_array = np.array(resized_img)
    input_img = np.expand_dims(resized_img_array, axis=0)
    result = model.predict_classes(input_img)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        label = predict(file_path)
        result = get_labels(label) 
        return result
    return None


if __name__ == '__main__':
    app.run()