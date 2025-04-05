from flask import Flask, render_template, Response, redirect, url_for, request, session
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
else:
    print("Camera opened")

# Video streaming function
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for model prediction (if model loading is enabled)
        resized_frame = cv2.resize(frame, (256, 256))  # Match your model's input
        # Temporarily skipping prediction logic
        cv2.putText(frame, f'Prediction: Model not loaded', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route for the main page with camera
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('Main.html')

# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Community page (formerly /community)
@app.route('/community')
def community():
    # Example data; replace with actual post data from a database or list.
    post = {
        'image': 'static/sample.jpg',
        'content': 'This is a sample post.'
    }
    return render_template('index.html', post=post)
# Profile page route (newly added)
@app.route('/profile')
def profile():
    return render_template('Profile.html')  # Assuming you have a Profile.html

# About page
@app.route('/about')
def about():
    return render_template('About.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get the values from the form
        username = request.form['username']
        password = request.form['password']

        # Hardcoded credentials check
        if username == 'admin@gmail.com' and password == '123':
            session['user_id'] = 'admin'  # Store the session data
            return redirect(url_for('index'))  # Redirect to the main page

        else:
            return 'Invalid credentials. Please try again.'  # Return error message for invalid login

    return render_template('Login.html')  # Display the login page on GET request

# Sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('SignUp.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove session
    return redirect(url_for('login'))  # Redirect to login page

@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'snapshot_{timestamp}.jpg'
        filepath = f'static/snapshots/{filename}'
        cv2.imwrite(filepath, frame)
        print(f"Saved snapshot to {filepath}")
    return redirect(url_for('index'))


@app.route('/submit_post', methods=['POST'])
def submit_post():
    # Retrieve form data (e.g., title, content, etc.)
    title = request.form.get('title')
    content = request.form.get('content')

    # Process and save the post data as needed
    # For now, we can simply print it or add it to a list/database
    print(f"New post received: Title: {title}, Content: {content}")

    # Redirect to the community page after processing
    return redirect(url_for('community'))
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
