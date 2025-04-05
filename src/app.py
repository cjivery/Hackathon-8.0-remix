from flask import Flask, render_template, Response, redirect, url_for, request, session
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from werkzeug.utils import secure_filename

os.makedirs('static/snapshots', exist_ok=True)


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the trained model from the specified path
model_path = r"C:\Users\sasuk\PycharmProjects\Hackathon-8.0-remix\plant_disease_model.h5"
model = load_model(model_path)

# Define your class names (update this list if needed)
class_names = [
    "Pepper_bell_Baterial_spot", "Pepper_bell_healthy",
    "Potato_early_blight", "Potato_healthy", "Potato_Late_blight",
    "Tomato_Target_Spot", "Tomato_Tomato_mosaic_virus", "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Bacterial_spot", "Tomato_Early_Blight", "Tomato_healthy", "Tomato_Late_Blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mites"
]

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Video streaming function with model prediction
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize & preprocess frame
        resized_frame = cv2.resize(frame, (256, 256))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image_array = img_to_array(rgb_frame)
        image_array = np.expand_dims(image_array, axis=0) / 255.0

        # Make prediction
        predictions = model.predict(image_array)
        print("Predictions:", predictions)  # Debug: print raw model output
        predicted_class = class_names[np.argmax(predictions)]

        # Encode frame for web stream
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")
            break

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess image
    img = cv2.imread(filepath)
    if img is None:
        return "Failed to load image", 400  # Check if the image loaded correctly
    print(f"Loaded image shape: {img.shape}")  # Debug: print image shape
    img = cv2.resize(img, (256, 256))
    print(f"Resized image shape: {img.shape}")  # Debug: print resized shape
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    print(f"Image array shape after preprocessing: {img_array.shape}")  # Debug: print final array shape

    # Predict
    predictions = model.predict(img_array)
    print(f"Raw Predictions: {predictions}")  # Debug: print raw model output

    predicted_class_idx = np.argmax(predictions)  # Get index of highest prediction
    predicted_class = class_names[predicted_class_idx]
    print(f"Predicted Class: {predicted_class}")  # Debug: print predicted class

    return render_template('Results.html', prediction=predicted_class, image_path=filepath)

# Route for the main page with camera feed
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get list of snapshot filenames
    snapshot_folder = 'static/snapshots'
    snapshots = os.listdir(snapshot_folder)
    snapshots.sort(reverse=True)  # Show latest first
    snapshot_urls = [f'{snapshot_folder}/{filename}' for filename in snapshots]

    return render_template('Main.html', snapshots=snapshot_urls)

# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Community page (serving index.html as community page)
@app.route('/community')
def community():
    return render_template('index.html')

# Profile page route
@app.route('/profile')
def profile():
    return render_template('Profile.html')

# About page route
@app.route('/about')
def about():
    return render_template('About.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Hardcoded credentials check
        if username == 'admin@gmail.com' and password == '123':
            session['user_id'] = 'admin'
            return redirect(url_for('index'))
        else:
            return 'Invalid credentials. Please try again.'

    return render_template('Login.html')

# Sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('SignUp.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

# Route to save a snapshot from the camera feed
@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    for _ in range(5):
        cap.read()  # flush buffer
    ret, frame = cap.read()

    if ret:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'snapshot_{timestamp}.jpg'
        filepath = f'static/snapshots/{filename}'
        cv2.imwrite(filepath, frame)
        print(f"Snapshot saved to {filepath}")
    else:
        print("Failed to capture snapshot.")

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
