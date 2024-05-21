from flask import Flask
from flask_cors import CORS
import cv2
from flask_ngrok import run_with_ngrok
from pyngrok import ngrok
from keras.models import load_model

app = Flask(__name__)
CORS(app)
# run_with_ngrok(app)

# Set up ngrok authentication token
# ngrok.set_auth_token("2gDk6gkcJSCa7YpuRI7xQPnzqQc_7cujt3BiEHW9HYTFZfMXH")

# Load the models and face classifiers
KNOWN_IMAGE_PATH = "app/model/WIN_20230528_23_10_59_Pro.jpg"  # Update path to your known image
face_classifier = cv2.CascadeClassifier('app/model/haarcascade_frontalface_default.xml')
classifier = load_model('app/model/model.h5')

from app import routes  # Import routes after initializing the app and models
