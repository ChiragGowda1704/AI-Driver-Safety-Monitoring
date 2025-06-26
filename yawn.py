import cv2
import dlib
import numpy as np
import pygame
import time
import os

pygame.mixer.init()

YAWN_THRESHOLD = 25  # Lip distance threshold for yawn detection
INITIAL_ALARM_YAWN_COUNT = 5  # Initial number of yawns to trigger alarm
ALARM_INCREMENT = 2  # Number of additional yawns to trigger alarm again
ALARM_SOUND_FILE = 'alarm.wav' 
ALARM_DURATION = 2  # Seconds to play the alarm

# Check if alarm sound file exists
if not os.path.exists(ALARM_SOUND_FILE):
    raise FileNotFoundError(f"Alarm sound file '{ALARM_SOUND_FILE}' not found in the project directory")

# Face detector and landmark predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"Shape predictor file '{PREDICTOR_PATH}' not found. Download it from dlib.net")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) != 1:
        return None
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[0, 1])  # Fixed numpy deprecation warning

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[0, 1])  # Fixed numpy deprecation warning

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks is None:
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

def play_alarm():
    try:
        pygame.mixer.music.load(ALARM_SOUND_FILE)
        pygame.mixer.music.play()
        return time.time()  # Return the start time
    except Exception as e:
        print(f"Error playing alarm: {e}")
        return None

def stop_alarm():
    pygame.mixer.music.stop()

# Main program
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

yawns = 0
yawn_status = False 
alarm_playing = False
next_alarm_threshold = INITIAL_ALARM_YAWN_COUNT
alarm_start_time = None

while True:
    ret, frame = cap.read()   
    if not ret:
        break
        
    image_landmarks, lip_distance = mouth_open(frame)
    
    prev_yawn_status = yawn_status  
    
    if lip_distance > YAWN_THRESHOLD:
        yawn_status = True 
        
        cv2.putText(frame, "Subject is Yawning", (50, 450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        output_text = f"Yawn Count: {yawns}"
        cv2.putText(frame, output_text, (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1
        # Check if we need to trigger the alarm
        if yawns >= next_alarm_threshold:
            alarm_start_time = play_alarm()
            if alarm_start_time:  # Only update if alarm started successfully
                alarm_playing = True
                next_alarm_threshold = yawns + ALARM_INCREMENT
    
    # Display alarm status if active
    if alarm_playing:
        cv2.putText(frame, "ALARM: Excessive Yawning!", (50, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        # Stop alarm after duration
        if alarm_start_time and (time.time() - alarm_start_time > ALARM_DURATION):
            stop_alarm()
            alarm_playing = False
    
    cv2.imshow('Live Landmarks', image_landmarks)
    cv2.imshow('Yawn Detection', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):  # 'q' key to exit
        break
    elif key == ord('r'):  # Reset counter with 'r' key
        yawns = 0
        next_alarm_threshold = INITIAL_ALARM_YAWN_COUNT
        if alarm_playing:
            stop_alarm()
            alarm_playing = False
        
cap.release()
cv2.destroyAllWindows()
stop_alarm()  # Ensure alarm stops when program ends