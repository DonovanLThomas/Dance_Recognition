import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import time
from picamera2 import Picamera2

model = tf.keras.models.load_model("new_model.h5")
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

reverse_label_map = {v: k for k, v in label_map.items()}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

sequence = []
SEQUENCE_LENGTH = 70

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("Don Tech")
picam2.start()

time.sleep(1)

while True:

    frame = picam2.capture_array()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        keypoints = np.array([[l.x, l.y, l.x, l.visibility] for l in landmarks]).flatten()
        sequence.append(keypoints)

        if len(sequence) >  SEQUENCE_LENGTH:
            sequence.pop(0)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data, verbose=0)
            predicted_label = reverse_label_map[np.argmax(prediction)]
            confidence = np.max(prediction)

            cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    
    cv2.imshow("Live Dance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
