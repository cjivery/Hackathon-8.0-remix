from flask import Flask, render_template, request, redirect, url_for
import subprocess
import threading

app = Flask(__name__)

# Simulate a simple blog data structure
posts = [
    {
        "id": 1,
        "title": "Welcome to My Blog!",
        "content": "This is a sample blog post. You can customize this however you like!",
        "image": "https://via.placeholder.com/600x300",
        "votes": 12,
        "replies": [
            {"text": "Great post! Learned a lot.", "timestamp": "2025-04-04 10:32 AM", "votes": 4},
            {"text": "Thanks for sharing!", "timestamp": "2025-04-04 11:00 AM", "votes": 2},
        ]
    }
]

@app.route('/')
def home():
    return render_template('index.html', posts=posts)

@app.route('/login')
def login():
    return render_template('Login.html')

@app.route('/signup')
def signup():
    return render_template('SignUp.html')

@app.route('/submit_post', methods=['POST'])
def submit_post():
    title = request.form['title']
    content = request.form['content']
    posts.append({
        "id": len(posts) + 1,
        "title": title,
        "content": content,
        "image": "https://via.placeholder.com/600x300",
        "votes": 0,
        "replies": []
    })
    return redirect(url_for('home'))

@app.route('/submit_reply/<int:post_id>', methods=['POST'])
def submit_reply(post_id):
    reply_text = request.form['reply_text']
    timestamp = "2025-04-04 11:30 AM"  # Static timestamp for now
    for post in posts:
        if post['id'] == post_id:
            post['replies'].append({"text": reply_text, "timestamp": timestamp, "votes": 0})
    return redirect(url_for('home'))

@app.route('/vote/<action>/<int:post_id>', methods=['POST'])
def vote(action, post_id):
    for post in posts:
        if post['id'] == post_id:
            if action == 'upvote':
                post['votes'] += 1
            elif action == 'downvote':
                post['votes'] -= 1
    return redirect(url_for('home'))

@app.route('/vote_reply/<action>/<int:post_id>/<int:reply_index>', methods=['POST'])
def vote_reply(action, post_id, reply_index):
    for post in posts:
        if post['id'] == post_id:
            if action == 'upvote':
                post['replies'][reply_index]['votes'] += 1
            elif action == 'downvote':
                post['replies'][reply_index]['votes'] -= 1
    return redirect(url_for('home'))
@app.route('/camera')
def run_camera():
    try:
        subprocess.Popen(['python', 'Camera_app.py'])
        return "<h2>Camera launched successfully. Check your webcam window.</h2><a href='/'>Back to Home</a>"
    except Exception as e:
        return f"<h2>Error launching camera: {e}</h2><a href='/'>Back to Home</a>"
def launch_camera_on_start():
    subprocess.Popen(['python', 'Camera_app.py'])


if __name__ == '__main__':
    threading.Thread(target=launch_camera_on_start).start()
    app.run(debug=True)
