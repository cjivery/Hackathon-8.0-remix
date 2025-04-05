from flask import Flask, render_template, Response, redirect, url_for, request, session
import cv2

app = Flask(__name__)

# Secret key for session management (required for session functionality)d
app.secret_key = 'your_secret_key'

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Video streaming function
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
        return redirect(url_for('login'))  # If the user is not logged in, redirect to login
    return render_template('Main.html')

# Video feed route for OpenCV streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Here you would handle user authentication (login logic)
        session['user_id'] = 'some_user_id'  # Save user ID in session
        return redirect(url_for('index'))  # Redirect to Main page after login
    return render_template('Login.html')

# Route for SignUp Page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Here you would handle user registration logic
        return redirect(url_for('login'))  # Redirect to login after sign-up
    return render_template('SignUp.html')

# Route to handle logout (redirecting to login)
@app.route('/logout')
def logout():
    # Clear session or any required session variables
    session.pop('user_id', None)  # Remove user_id from session
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
