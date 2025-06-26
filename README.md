# 🚗 AI-Powered Driver Safety Monitoring System

## 🔍 Problem Statement

In recent years, automotive innovation has focused on enhancing driver safety through smart monitoring systems. Understanding a driver’s posture, eye movements, facial expressions, and hand gestures can help determine their level of alertness, focus, and mood.

Advanced driver monitoring systems (DMS) aim to:
- Detect signs of fatigue, distraction, and drowsiness
- Monitor driver activities like phone usage or eating
- Enable smart, responsive features like alert generation, media control, and emergency assistance

Tomorrow’s cars won’t just drive better — they’ll also understand you better.

---

## 💡 Solution Overview

This system combines **Computer Vision**, **Heart Rate Monitoring**, and **Driving Behavior Classification** to intelligently assess driver safety and behavior in real-time.

### 👁️ Computer Vision Modules
1. **Driver Identification** – Auto-restore driver preferences.
2. **Activity Recognition**
   - Deep learning-based recognition of risky behaviors like:
     - Talking on the phone
     - Eating while driving
   - Triggers alerts to raise awareness.
3. **Drowsiness & Distraction Detection**
   - Detects yawning, eye closure, and loss of attention.
   - Real-time alerts using visual + audio feedback.
   - Eye-based Human-Machine Interface (HMI) control.
4. **Hand Gesture Control**
   - Neural networks recognize hand gestures for controlling car features like volume or AC.
5. **Heart Rate Monitoring**
   - A grip sensor in the steering wheel monitors heart rate.
   - Alerts triggered for fatigue or irregular heartbeat.

---

## 🚦 Driving Style Classification (AI-Based)

An AI model analyzes driving patterns and behaviors to detect **aggressive driving**, using:

### 🔸 Input Criteria
- Sudden acceleration or braking
- Sharp turns
- RPM patterns
- Red light jumps
- Tailgating or honking
- Wrong-side overtakes

### 🔸 Fuzzy Logic System
1. **Fuzzification** – Converts sensor inputs into fuzzy variables.
2. **Rules Evaluation** – Applies logic rules to assess risk levels.
3. **Defuzzification** – Outputs clear classification of driving style.

### 🧪 Dynamic Thresholds
Thresholds (e.g. for speed or acceleration) adapt based on road type — city, highway, or national route.

---

## 🆕 Novelty & Innovation

Unlike traditional DMS systems that rely only on image processing, our solution is a **multi-modal approach** combining:
- Visual data
- Driving pattern analysis
- Bio-sensor inputs (heart rate)

This results in more **accurate**, **intelligent**, and **realistic** safety monitoring.

---

## 🛠️ Implementation Plan

### 1. Data Acquisition
- Webcam monitors face and posture
- Grip-based heart rate sensor embedded in steering wheel

### 2. Preprocessing
- Sensor and video data cleaned and normalized

### 3. Feature Extraction
- Deep learning models extract eye, face, and mouth landmarks
- Audio/gesture/movement features analyzed

### 4. Classification
- Fuzzy logic system classifies driver state: Alert, Drowsy, Distracted, Joyful, etc.

---

## 🚀 Getting Started

### 📥 Clone the Repository
```
git clone https://github.com/ChiragGowda1704/AI-Driver-Safety-Monitoring.git
