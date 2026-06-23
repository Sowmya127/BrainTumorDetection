from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import sqlite3

app = Flask(__name__)
model = load_model('models/model.keras')
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DATABASE = 'predictions.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            patient_age INTEGER,
            medical_history TEXT,
            image_path TEXT,
            prediction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.close()

init_db()

def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]
    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)
            result, confidence = predict_tumor(file_location)
            patient_name = request.form.get('patient_name')
            patient_age = request.form.get('patient_age')
            medical_history = request.form.get('medical_history')
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO predictions (patient_name, patient_age, medical_history, image_path, prediction)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_name, patient_age, medical_history, file_location, result.split(',')[0])) # Store only the tumor type.
            conn.commit()
            conn.close()
            return render_template('index.html', result=result, file_path=f'/uploads/{file.filename}')
    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def history():
    conn = get_db_connection()
    predictions = conn.execute('SELECT patient_name, patient_age, medical_history, image_path, prediction, timestamp FROM predictions ORDER BY timestamp DESC').fetchall()
    conn.close()
    return render_template('history.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)