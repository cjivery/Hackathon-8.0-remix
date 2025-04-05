from flask import Flask, render_template, Response, redirect, url_for, request, session
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

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
            break

        # Resize frame to match the model's input size
        resized_frame = cv2.resize(frame, (256, 256))
        image_array = img_to_array(resized_frame)
        image_array = np.expand_dims(image_array, axis=0) / 255.0

        # Predict using the model
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions)]

        # Overlay the prediction on the original frame
        cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route for the main page with camera feed
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('Main.html')

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
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'snapshot_{timestamp}.jpg'
        filepath = f'static/snapshots/{filename}'
        cv2.imwrite(filepath, frame)
        print(f"Snapshot saved to {filepath}")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
