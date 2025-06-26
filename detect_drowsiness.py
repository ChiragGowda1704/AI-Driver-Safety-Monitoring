from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import pygame
import os

def sound_alarm(path):
    """Play an alarm sound using pygame for better control"""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing alarm: {e}")

def stop_alarm():
    """Stop the currently playing alarm"""
    pygame.mixer.music.stop()

def eye_aspect_ratio(eye):
    """Calculate eye aspect ratio (EAR)"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculate mouth aspect ratio (MAR) for yawn detection"""
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def head_position(shape, frame_width, frame_height):
    """Estimate head position using facial landmarks"""
    nose = shape[30]  # Nose tip landmark
    x_ratio = nose[0] / frame_width
    y_ratio = nose[1] / frame_height
    
    # Determine head position
    if x_ratio < 0.35:
        h_position = "LEFT"
    elif x_ratio > 0.65:
        h_position = "RIGHT"
    else:
        h_position = "CENTER"
    
    return h_position, x_ratio, y_ratio

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="alarm.wav",
                help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
MAR_THRESH = 0.5  # Mouth aspect ratio threshold for yawn
YAWN_CONSEC_FRAMES = 20
HEAD_POSITION_ALERT_FRAMES = 30

# Initialize counters and status flags
COUNTER = 0
ALARM_ON = False
yawn_counter = 0
head_position_counter = 0
blink_count = 0
start_time = time.time()

# Initialize dlib's face detector and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Facial landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Start video stream
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# Create fullscreen window
cv2.namedWindow("Driver Drowsiness Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Driver Drowsiness Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Main loop
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)  # Larger frame for fullscreen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # Get frame dimensions
    (h, w) = frame.shape[:2]
    
    # Loop over face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract eye and mouth regions
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # Calculate eye aspect ratio
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Calculate mouth aspect ratio
        mar = mouth_aspect_ratio(mouth)

        # Estimate head position
        h_position, x_ratio, y_ratio = head_position(shape, w, h)

        # Visualize eyes and mouth
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)

        # Check for drowsiness (eye closure)
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Detect blink (short eye closure)
            if 1 < COUNTER < EYE_AR_CONSEC_FRAMES:
                blink_count += 1
            
            COUNTER = 0
            ALARM_ON = False

        # Check for yawning
        if mar > MAR_THRESH:
            yawn_counter += 1
            
            if yawn_counter >= YAWN_CONSEC_FRAMES:
                cv2.putText(frame, "YAWN DETECTED!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if not ALARM_ON:
                    ALARM_ON = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()
        else:
            yawn_counter = 0

        # Check for head position
        if h_position != "CENTER":
            head_position_counter += 1
            
            if head_position_counter >= HEAD_POSITION_ALERT_FRAMES:
                cv2.putText(frame, f"HEAD {h_position}!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            head_position_counter = 0

        # Display information
        elapsed_time = time.time() - start_time
        blink_rate = blink_count / (elapsed_time / 60)  # blinks per minute
        
        cv2.putText(frame, f"EAR: {ear:.2f}", (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (w - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_count} ({blink_rate:.1f}/min)", (w - 250, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Head: {h_position}", (w - 150, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display "No face detected" if no faces found
    if len(rects) == 0:
        cv2.putText(frame, "NO FACE DETECTED", (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame in fullscreen
    cv2.imshow("Driver Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Reset counters with 'r' key
    if key == ord("r"):
        blink_count = 0
        start_time = time.time()
        if ALARM_ON:
            stop_alarm()
            ALARM_ON = False

    # Exit on 'q' key
    if key == ord("q"):
        break

# Cleanup
stop_alarm()
cv2.destroyAllWindows()
vs.stop()