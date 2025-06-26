# 🚗 AI-Based Driver Safety Monitoring System

A real-time driver monitoring system built using deep learning and computer vision to enhance road safety by detecting drowsiness, yawning, eye blinks, and mood of the driver. The system triggers an alarm when signs of fatigue or distraction are detected, ensuring the driver remains alert on the road.

---

## 🧠 Core Features

### 1. **Face Detection**
- Utilizes `dlib`'s facial landmark detector to detect and track the driver's face in real time.
- Enables the foundation for further analysis like eye, mouth, and emotion detection.

### 2. **Eye Blink Detection**
- Tracks the driver’s eye movements using facial landmarks.
- Counts the number of blinks per frame to monitor signs of fatigue.
- If abnormal blinking patterns are detected, the system raises awareness.

### 3. **Drowsiness Detection**
- Detects when the driver’s eyes remain closed for a continuous number of frames.
- Triggers a **warning alarm** when drowsiness is detected.
- Helps prevent micro-sleep events while driving.

### 4. **Yawn Detection**
- Analyzes mouth landmarks to identify yawning behavior.
- Plays an **alert sound** if a yawn is detected, prompting the driver to stay alert.

### 5. **Mood Detection**
- Uses a trained deep learning model (TensorFlow) to classify driver emotion.
- Can detect basic moods such as happy, sad, angry, or neutral.
- Helps understand emotional state which could affect driving focus.

---

## 🚀 Getting Started

### 🔁 Clone the repository

git clone https://github.com/ChiragGowda1704/AI-Driver-Safety-Monitoring.git
## 📂 Required Files & Setup

Before running the project, make sure you have the following essential files:

### 🔸 Files to Download:
- **Pre-trained Emotion Detection Model** – Used for real-time mood classification.
- **Haarcascade XML Files** – For basic face and feature detection using OpenCV.
- **Dlib Shape Predictor**
  - File: `shape_predictor_68_face_landmarks.dat`
  - Purpose: Accurately maps 68 facial landmarks for detecting eyes, mouth, and other key features.

📁 **Place all downloaded files directly in the root directory of the project** so the scripts can access them without path issues.

---

## 📦 Dependency Installation

Install all required Python libraries using pip:

```
pip install opencv-python dlib keras imutils numpy pygame tensorflow
```
## **🧪 How to Run the Modules**
Each module is designed for a specific safety feature and can be run independently:

▶️ Drowsiness Detection
Detects if the driver’s eyes are closed for too long. Triggers an alarm to prevent micro-sleep.
python drowsiness_detection.py

▶️ Yawn Detection
Analyzes mouth landmarks to detect yawning. If a yawn is detected, a warning alarm is triggered to keep the driver alert.
python yawn.py

▶️ Mood Recognition
Uses a trained TensorFlow model to classify the driver's current mood (happy, sad, neutral, etc.) in real-time.
python mood_recognition.py

## 📬 Contact

Feel free to connect with me or reach out if you have any questions, feedback, or ideas for improvements:

🔗 [LinkedIn – Chirag Gowda](https://www.linkedin.com/in/chiraggowda17/)

🐞 Found a bug or want to suggest a feature?  
Raise an [issue](../../issues) or start a discussion!

---

> **🚗 Drive Safe. 💻 Code Smart. 🛡️ Stay Alert.**

