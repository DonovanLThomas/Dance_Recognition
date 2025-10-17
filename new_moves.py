import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
import time
import pickle

SEQUENCE_LENGTH = 70

#Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


#Pi cam Set up
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("Don Tech")
picam2.start()

time.sleep(1)

#State Tracking
current_label = None
recording = False
current_sequence = []
all_sequences = []

while True:

    frame = picam2.capture_array()

    #frame conversion for compatibility for mediapipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results = pose.process(rgb_frame)

    display_frame = frame.copy()
    
    #Key is for labeling the session so c is chicago dancing j is for jersey dancing and h is for hous     ton dancing
    key = cv.waitKey(1) & 0xFF

    if key in [ord('j'), ord('c'), ord('h')]:
        current_label = chr(key)
        current_sequence = []
        recording = True
        print(f"[INFO] Starting recording for label: {current_label}")
    elif key == ord('q'):
        break

    if recording and results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

        current_sequence.append(landmarks)

        if len(current_sequence) == SEQUENCE_LENGTH:
            all_sequences.append((current_sequence.copy(), current_label))
            print(f"[INFO] Saved sequence for '{current_label}'")
            recording = False
            current_label = None
    
    if current_label:
        cv.putText(display_frame, f"Recording: {current_label}", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    cv.imshow('Live Stream', display_frame)


cv.destroyAllWindows()
picam2.close()

with open("poses.pkl", "wb") as f:
    pickle.dump(all_sequences, f)

print(f"[INFO] Saved {len(all_sequences)} sequences to 'pose_sequences'")
