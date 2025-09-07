import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = os.path.join('models', 'cattle_breed_model.h5')
CLASSES_PATH = os.path.join('models', 'classes', 'breeds.json')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load classes
with open(CLASSES_PATH, 'r') as f:
    CLASS_NAMES = json.load(f)

print("Class names loaded successfully!")

# Load the full model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid file"}), 400

    # Save file
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    preds = model.predict(img_array)
    index = int(np.argmax(preds[0]))
    confidence = float(preds[0][index] * 100)

    breed_name = CLASS_NAMES[index]['breed']
    animal_type = CLASS_NAMES[index]['animal']

    # Clean up if you want
    # os.remove(filepath)

    # Return result
    return render_template("result.html",
                           filename=filename,
                           breed_name=breed_name,
                           animal_type=animal_type,
                           confidence=round(confidence, 2))

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    app.run(debug=True)
