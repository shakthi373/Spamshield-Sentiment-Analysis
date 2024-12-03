import os
from flask import Flask, flash, request, render_template, session, redirect, url_for,jsonify
import sqlite3
import nltk
import re
import joblib
import googleapiclient.discovery
import pandas as pd
from textblob import TextBlob
from werkzeug.security import check_password_hash
from datetime import timedelta
from dotenv import load_dotenv
from lime.lime_text import LimeTextExplainer
from sklearn.utils.extmath import softmax

load_dotenv()
api_key = os.getenv('YOUTUBE_API_KEY')

# Download NLTK resources
nltk.download('punkt')

app = Flask(__name__)

# Set the secret key for session management
app.secret_key = os.getenv('FLASK_SECRET_KEY','default_secret_key' ) 
app.permanent_session_lifetime = timedelta(minutes=15)
app.config['SESSION_COOKIE_SECURE'] = True
#login page

DATABASE = 'database.db'

def create_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  name TEXT,
                  email TEXT, 
                  mobile INTEGER, 
                  username TEXT UNIQUE, 
                  password TEXT)''')
    conn.commit()
    conn.close()

from werkzeug.security import generate_password_hash

def insert_user(name, email, mobile, username, password):
    # Hash the password before storing it
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    sql_query = "INSERT INTO users (name, email, mobile, username, password) VALUES (?, ?, ?, ?, ?)"
    params = (name, email, mobile, username, hashed_password)
    c.execute(sql_query, params)
    conn.commit()
    conn.close()



def get_user(username):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form['name'] 
        email = request.form['email']
        mobile = request.form['mobile']
        username = request.form['username']
        password = request.form['password']
        
        if get_user(username):
            message = "User already exists!"
            return render_template('signup.html', message=message)
        insert_user(name, email, mobile, username, password)
        message = "Account successfully created"
        return render_template('signup.html', message=message)
    return render_template('signup.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)

        if user and check_password_hash(user[5], password):  # user[5] is the password field in the DB
            session['username'] = username
            return redirect(url_for('upload'))
        return render_template('login.html', message="Invalid username or password!")
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    session.pop('username', None)
    return render_template("landing.html")


@app.route("/")
def home():
    return render_template("landing.html")

@app.route('/landing')
def landing():
    return render_template("landing.html")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')

        if not name or not email or not subject or not message:
            flash('All fields are required!')
            return redirect(url_for('contactus'))

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS user_query 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  name TEXT,
                  email TEXT, 
                  subject TEXT, 
                  message TEXT)''')
        cursor.execute("INSERT INTO user_query (name, email, subject, message) VALUES (?, ?, ?, ?)",
                       (name, email, subject, message))
        conn.commit()
        conn.close()

        flash('Thank you for contacting us!', 'success')
        return redirect(url_for('contactus'))

    return render_template('contactus.html')


#login completed


@app.route("/")
def homepage():
    return render_template('landing.html')



def extract_video_id(url):
    """
    Extract the YouTube video ID from a URL.
    Supports both short and long YouTube URLs.
    """
    # Regex pattern to extract video ID
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.match(pattern, url)
    return match.group(1) if match else None

def get_video_comments(video_id, api_key):
    # Build the YouTube service
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    # Initialize variables
    comments = []
    next_page_token = None

    while True:
        # Request to get comments
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100,  # Max is 100
            textFormat="plainText"
        )
        response = request.execute()

        # Parse the response
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        # Check for next page
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_non_ascii(text):
    # Remove non-ASCII characters
    return ''.join(char for char in text if ord(char) < 128)

def remove_digits(text):
    # Remove numeric digits
    return re.sub(r'\d+', '', text)

def remove_special_characters(text):
    # Remove special characters except whitespace
    return re.sub(r'[^\w\s]', '', text)

def normalize_case(text):
    # Normalize text to lowercase
    return text.lower()

def clean_text(text):
    # Remove URLs
    text = remove_urls(text)
    # Remove non-ASCII characters
    text = remove_non_ascii(text)
    # Remove numeric digits
    text = remove_digits(text)
    # Remove special characters except whitespace
    text = remove_special_characters(text)
    # Normalize case
    text = normalize_case(text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

spam_explainer = LimeTextExplainer(class_names=["Not Spam", "Spam"])
sentiment_explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])
model = joblib.load('spam_detection_model.pkl')
sentiment_model = joblib.load('sentiment_detection_model.pkl')
def explain_spam(comment):
    """
    Generate LIME explanations for spam classification.
    """
    explanation = spam_explainer.explain_instance(
        comment,
        model.predict_proba,
        num_features=5,
        labels=(0, 1)
    )
    return explanation.as_list()


def explain_sentiment(comment):
    """
    Generate LIME explanations for sentiment analysis.
    """
    explanation = sentiment_explainer.explain_instance(
        comment,
        lambda x: softmax(sentiment_model.decision_function(x)),
        num_features=5
    )
    return explanation.as_list()
emotion_model = joblib.load('emotion_classifier_pipe_lr.pkl')

def predict_emotion(text):
    """
    Predict the emotion of a given text using the emotion model.
    """
    prediction = emotion_model.predict([text])[0]
    probability = emotion_model.predict_proba([text])[0]
    return prediction, probability

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion_api():
    """
    API endpoint for emotion prediction.
    """
    data = request.json
    text = data['text']
    prediction, probability = predict_emotion(text)
    return jsonify({
        'emotion': prediction,
        'probabilities': probability.tolist()
    })
@app.route('/explain_spam', methods=['POST'])
def explain_spam_api():
    data = request.json
    comment = data['text']
    explanation = explain_spam(comment)
    return jsonify({'spam_explanation': explanation})


@app.route('/explain_sentiment', methods=['POST'])
def explain_sentiment_api():
    data = request.json
    comment = data['text']
    explanation = explain_sentiment(comment)
    return jsonify({'sentiment_explanation': explanation}) 
@app.route("/upload", methods=["GET", "POST"])
def upload():
    
    if request.method == "POST":
        url = request.form.get('search')

        # Validate if the URL starts with http or https
        if not (url.startswith("http://") or url.startswith("https://")):
            return render_template('prediction.html', error="Invalid YouTube URL")
   
        video_id = extract_video_id(url)  # Extract the video ID from the URL
        if video_id:
            comments = get_video_comments(video_id, api_key)  # Fetch all comments from the YouTube API
            original_comments = comments.copy()
            total_fetched_comments = len(comments)  # Track total fetched comments

            # Clean and filter comments
            cleaned_comments = [clean_text(comment) for comment in comments]

            filtered_comments = [comment for comment in cleaned_comments if len(comment) > 5]
            total_filtered_comments = len(filtered_comments)  # Track the number of comments left after filtering

            df_comments = pd.DataFrame({
                'comment': filtered_comments,
                'original_comment': original_comments[:len(filtered_comments)]
            })
                
            # Spam prediction and explanations
            spam_predictions = model.predict(df_comments['comment'])
            df_comments['spam_prediction'] = spam_predictions
            non_spam_comments = df_comments[df_comments['spam_prediction'] == 0].reset_index(drop=True)

            # Sentiment analysis and explanations
            sentiment_predictions = sentiment_model.predict(non_spam_comments['comment'])
            non_spam_comments['sentiment_prediction'] = sentiment_predictions
            # non_spam_comments['sentiment_explanation'] = non_spam_comments['comment'].apply(get_sentiment_explanation)
                    
        else:
            return render_template('prediction.html', error="Invalid YouTube URL")

        # new_predictions = model.predict(df_comments['comment'])
        # Calculate total non-spam comments
        total_comments = len(non_spam_comments)

        # Count the number of each sentiment type
        sentiment_counts = non_spam_comments['sentiment_prediction'].value_counts()

        positive_percentage = round((sentiment_counts.get(2, 0) / total_comments) * 100, 2)
        neutral_percentage = round((sentiment_counts.get(1, 0) / total_comments) * 100, 2)
        negative_percentage = round((sentiment_counts.get(0, 0) / total_comments) * 100, 2)

        # Debugging prints (can be removed later)
        print(f"Youtube Video URL ID: {video_id}")
        print(f"Total Comments Fetched: {total_fetched_comments}")
        print(f"Total Filtered Comments: {total_filtered_comments}")
        print(f"Total Non-Spam Comments: {total_comments}")
        print(f"Positive Comments: {positive_percentage:.2f}%")
        print(f"Neutral Comments: {neutral_percentage:.2f}%")
        print(f"Negative Comments: {negative_percentage:.2f}%")

        # Render the template with all the required information
        return render_template(
           'prediction.html', 
            url=url, 
            video_id=video_id, 
            total_comments_fetched=total_fetched_comments,
            total_comments_filtered=total_filtered_comments,
            total_comments=total_comments,
            positive_percentage=positive_percentage, 
            neutral_percentage=neutral_percentage, 
            negative_percentage=negative_percentage,
            comments_data=non_spam_comments.to_dict(orient='records')
        )

    return render_template('prediction.html')
@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    
    app.run(debug=True)
