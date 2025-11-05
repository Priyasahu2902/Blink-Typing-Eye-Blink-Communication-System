# 👁️ Blink Typing – Eye-Blink Communication System

An **accessible AI-powered typing system** that allows users to type and communicate using only **eye blinks and head movements** — designed especially for individuals with motor disabilities.  

---

## 🚀 Project Overview
**Blink Typing (BlinkTalk)** is an innovative system that detects **short and long blinks**, **eye gaze**, and **head gestures** using real-time video input.  
The system translates these blinks into letters, words, and speech — enabling hands-free, voice-assisted communication.

---

## 🧠 Features
- 💤 **Short / Long Blinks for Letters**
- 🕐 **Space (2 eyes closed for 2s)**  
- ❌ **Delete (eyes closed for 4s)**  
- 🗣️ **Speak by looking LEFT or RIGHT**
- 🙅 **Cancel via head-shake gesture**
- 👁️ **Live video preview with status indicators**
- 🔊 **Text-to-Speech output**
- ⚙️ **Manual calibration (‘C’ key)** for accurate detection

---

## 📂 Project Structure
Blink_App/
│
├── static/
│ └── style.css # Frontend styling
│
├── templates/
│ └── index.html # Main UI page
│
├── blink_backend.py # Flask + OpenCV backend logic
├── requirements.txt # Dependencies
└── README.md # Documentation

## 🧩 Tech Stack
| Component | Technology |
|------------|-------------|
| **Frontend** | HTML, CSS (Flask Templates) |
| **Backend** | Python (Flask Framework) |
| **Computer Vision** | OpenCV, MediaPipe |
| **Speech Output** | pyttsx3 |
| **Data Handling** | NumPy |

---

## How It Works

1.Webcam captures your eyes and face using MediaPipe.
2.Calculates the Eye Aspect Ratio (EAR) to detect blinks.
3.Blink patterns (short/long) are mapped to letters.
4.Text appears live on the screen.
5.Looking right triggers the text-to-speech output.
