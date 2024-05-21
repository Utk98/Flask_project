from flask import request, jsonify
import cv2
import numpy as np
import face_recognition
from keras.preprocessing.image import img_to_array
from collections import Counter
from app import app, KNOWN_IMAGE_PATH, face_classifier, classifier

def emotion_fdetect(video_path):
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    cap = cv2.VideoCapture(video_path)
    emotion_counter = Counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                emotion_counter[label] += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    most_common_emotion = emotion_counter.most_common(1)
    if most_common_emotion:
        return most_common_emotion[0][0]

def detect_person_match(video_path):
    known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "Video Capture Error"

    face_locations = face_recognition.face_locations(frame)
    face_count = len(face_locations)
    if face_count == 0:
        cap.release()
        return "No Face Detected"
    if face_count > 1:
        cap.release()
        return "More than One Face Detected"

    face_encodings = face_recognition.face_encodings(frame, face_locations)
    match = face_recognition.compare_faces([known_encoding], face_encodings[0])
    if match[0]:
        neck_bending = detect_neck_bending(frame, face_locations[0])
        cap.release()
        return "Neck Movement" if neck_bending else "Match"
    else:
        cap.release()
        return "Not Match"

def detect_neck_bending(frame, face_location):
    face_landmarks = face_recognition.face_landmarks(frame, [face_location])[0]
    top_nose = face_landmarks['nose_bridge'][0]
    bottom_nose = face_landmarks['nose_tip'][0]
    top_chin = face_landmarks['chin'][8]
    bottom_chin = face_landmarks['chin'][0]

    neck_vector = np.array(bottom_chin) - np.array(top_chin)
    face_vector = np.array(bottom_nose) - np.array(top_nose)

    angle = np.degrees(np.arccos(np.dot(neck_vector, face_vector) /
                                  (np.linalg.norm(neck_vector) * np.linalg.norm(face_vector))))

    return angle > 130 or angle < 125

@app.route('/match_person', methods=['POST'])
def match_person():
    if 'video' not in request.files:
        return jsonify({'error': 'Missing video file'})

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'})

    video_path = 'temp_video.mp4'
    video_file.save(video_path)

    result = detect_person_match(video_path)
    emotion = emotion_fdetect(video_path)

    return jsonify({'result': result, 'emotion': emotion})
