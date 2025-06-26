from __future__ import division
import dlib
from imutils import face_utils
import cv2
import numpy as np
from scipy.spatial import distance as dist
import pygame
import time
import os

# Initialize pygame for sound
pygame.mixer.init()

# Constants
EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 20  # Number of consecutive frames for drowsiness detection
ALARM_SOUND_FILE = 'alarm.wav'  # Alarm sound file
ALARM_DURATION = 3  # Seconds to play alarm
DROWSINESS_RESET_FRAMES = 50  # Frames of open eyes to reset drowsiness counter

# Check if alarm sound file exists
if not os.path.exists(ALARM_SOUND_FILE):
    raise FileNotFoundError(f"Alarm sound file '{ALARM_SOUND_FILE}' not found")

# Load face detector and shape predictor
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def play_alarm():
    try:
        pygame.mixer.music.load(ALARM_SOUND_FILE)
        pygame.mixer.music.play()
        return time.time()
    except Exception as e:
        print(f"Error playing alarm: {e}")
        return None

def stop_alarm():
    pygame.mixer.music.stop()

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (width, height), interpolation)
        return resized

# Initialize camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise IOError("Cannot open webcam")

# Set fullscreen mode
cv2.namedWindow("Drowsiness Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Drowsiness Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize variables
total = 0
alarm_on = False
alarm_start_time = None
reset_counter = 0
blink_count = 0
start_time = time.time()

while True:
    ret, frame = camera.read()
    if not ret:
        print('Failed to capture frame from camera')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)

    # Detect faces
    dets = detector(frame_resized, 1)
    
    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = face_utils.shape_to_np(shape)
            
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear > EYE_AR_THRESH:
                reset_counter += 1
                if reset_counter >= DROWSINESS_RESET_FRAMES:
                    total = 0
                    reset_counter = 0
                
                cv2.putText(frame, "Eyes Open", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Detect blink (transition from closed to open)
                if total > 1:
                    blink_count += 1
            else:
                total += 1
                reset_counter = 0
                cv2.putText(frame, "Eyes Closed", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check for drowsiness
            if total >= EYE_AR_CONSEC_FRAMES:
                if not alarm_on:
                    alarm_start_time = play_alarm()
                    alarm_on = True
                cv2.putText(frame, "DROWSINESS ALERT!", (250, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                if alarm_on and alarm_start_time and (time.time() - alarm_start_time > ALARM_DURATION):
                    stop_alarm()
                    alarm_on = False

            for (x, y) in shape:
                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 2, (0, 255, 0), -1)

    # Calculate and display statistics
    elapsed_time = time.time() - start_time
    blink_rate = blink_count / (elapsed_time / 60) if elapsed_time > 0 else 0  # blinks per minute
    
    # Display information
    cv2.putText(frame, f"Frame Count: {total}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}/min", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}" if 'ear' in locals() else "EAR: N/A", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display warning if no face detected
    if len(dets) == 0:
        cv2.putText(frame, "No face detected!", (150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset counters
        total = 0
        blink_count = 0
        start_time = time.time()
        if alarm_on:
            stop_alarm()
            alarm_on = False

# Cleanup
stop_alarm()
camera.release()
cv2.destroyAllWindows()