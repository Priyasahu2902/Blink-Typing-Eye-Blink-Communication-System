"""
BlinkTalk – Final Version
-------------------------
Accessible Eye-Blink Typing System
Features:
- Short/Long blinks for letters
- Space (2s eyes closed)
- Delete (4s eyes closed)
- Speak by looking LEFT or RIGHT
- Head-shake cancel
- Live preview
- Speaking indicator on UI
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from flask import Flask, jsonify, request, Response, render_template
from flask_cors import CORS
from threading import Thread
import pyttsx3

# ---------- CONFIG ----------
EYE_AR_CLOSE_THRESH = 0.23
MIN_SHORT, MAX_SHORT = 0.15, 0.55
MIN_LONG, MAX_LONG = 0.55, 1.6
SPACE_HOLD = 2.0
DELETE_HOLD = 4.0
PATTERN_TIMEOUT = 2.5
DOUBLE_BLINK_CANCEL_MAX = 0.35
BLINK_COOLDOWN = 0.25
SMOOTH_FRAMES = 4
RIGHT_GAZE_HOLD = 1.5
HEAD_MOVE_CANCEL = 30
# ----------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

engine = pyttsx3.init()
app = Flask(__name__)
CORS(app)

status_data = {
    "ear": 0.0,
    "pattern": "-",
    "preview": "-",
    "final_text": "",
    "speaking": False,
    "status": "Ready"
}

cap = cv2.VideoCapture(0)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

BLINK_DICT = {
    "S": "A", "L": "B", "SS": "C", "SL": "D", "LS": "E", "LL": "F",
    "SSS": "G", "SSL": "H", "SLS": "I", "LSS": "J", "LSL": "K",
    "SLL": "L", "LLL": "M", "SSSS": "N", "SSSL": "O", "SSLS": "P",
    "SLLS": "Q", "SLSL": "R", "LSLS": "S", "LLSS": "T", "SLSLS": "U",
    "LSLSL": "V", "SSSLL": "W", "LSSSL": "X", "SLLSS": "Y", "LLLSS": "Z"
}


def ear_from_landmarks(lm, eye_idx, img_w, img_h):
    pts = [(int(lm[i].x * img_w), int(lm[i].y * img_h)) for i in eye_idx]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0.0


def eye_center(lm, eye_idx, img_w, img_h):
    xs = [lm[i].x * img_w for i in eye_idx]
    ys = [lm[i].y * img_h for i in eye_idx]
    return (np.mean(xs), np.mean(ys))


def classify_blink(duration):
    if MIN_SHORT <= duration <= MAX_SHORT:
        return "S"
    elif MIN_LONG <= duration <= MAX_LONG:
        return "L"
    elif duration >= DELETE_HOLD:
        return "DELETE"
    elif duration >= SPACE_HOLD:
        return "SPACE"
    return None


def decode_sequence(seq):
    s = ''.join(seq)
    return BLINK_DICT.get(s, '?' if s else "-")


# ---------- STATE ----------
blink_sequence = []
ear_history = []
last_blink_time = None
blink_start_time = None
last_two_blinks = []
final_text = ""
predicted_letter = "-"
prev_center_x, prev_center_y = None, None
baseline_center_x, baseline_center_y = None, None
right_gaze_start = None
speaking = False
# ---------------------------


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status', methods=['GET'])
def status():
    return jsonify(status_data)


@app.route('/command', methods=['POST'])
def command():
    global final_text
    cmd = request.json.get('command')
    if cmd == "clear":
        final_text = ""
    return jsonify({"ok": True})


@app.route('/video_feed')
def video_feed():
    """Live webcam feed with blink logic"""
    def gen_frames():
        global blink_sequence, last_blink_time, blink_start_time
        global last_two_blinks, final_text, predicted_letter
        global prev_center_x, prev_center_y, baseline_center_x, baseline_center_y
        global right_gaze_start, speaking

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            ear = None

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                left_ear = ear_from_landmarks(lm, LEFT_EYE_IDX, w, h)
                right_ear = ear_from_landmarks(lm, RIGHT_EYE_IDX, w, h)
                ear = (left_ear + right_ear) / 2.0
                ear_history.append(ear)
                if len(ear_history) > SMOOTH_FRAMES:
                    ear_history.pop(0)
                ear = np.mean(ear_history)

                left_center = eye_center(lm, LEFT_EYE_IDX, w, h)
                right_center = eye_center(lm, RIGHT_EYE_IDX, w, h)
                face_center_x = (left_center[0] + right_center[0]) / 2
                face_center_y = (left_center[1] + right_center[1]) / 2

                if baseline_center_x is None:
                    baseline_center_x = face_center_x
                    baseline_center_y = face_center_y

                # --- Head shake cancel (left-right or up-down) ---
                if prev_center_x is not None and prev_center_y is not None:
                    dx = abs(face_center_x - prev_center_x)
                    dy = abs(face_center_y - prev_center_y)
                    if dx > HEAD_MOVE_CANCEL or dy > HEAD_MOVE_CANCEL:
                        blink_sequence.clear()
                        predicted_letter = "-"
                prev_center_x, prev_center_y = face_center_x, face_center_y

                # --- Left or Right gaze (Speak) ---
                if abs(face_center_x - baseline_center_x) > 40:
                    if right_gaze_start is None:
                        right_gaze_start = time.time()
                    elif time.time() - right_gaze_start >= RIGHT_GAZE_HOLD:
                        if final_text.strip() and not speaking:
                            speaking = True
                            status_data["status"] = "Speaking..."
                            engine.say(final_text.strip())
                            engine.runAndWait()
                            speaking = False
                            status_data["status"] = "Ready"
                            right_gaze_start = None
                else:
                    right_gaze_start = None

            now = time.time()

            # --- Blink detection ---
            if ear is not None and ear < EYE_AR_CLOSE_THRESH:
                if blink_start_time is None:
                    blink_start_time = now
            else:
                if blink_start_time is not None:
                    duration = now - blink_start_time
                    blink_start_time = None
                    typ = classify_blink(duration)

                    if typ == "SPACE":
                        final_text += " "
                        blink_sequence.clear()
                        predicted_letter = "-"
                    elif typ == "DELETE":
                        if final_text:
                            final_text = final_text[:-1]
                        blink_sequence.clear()
                        predicted_letter = "-"
                    elif typ in ("S", "L"):
                        blink_sequence.append(typ)
                        predicted_letter = decode_sequence(blink_sequence)
                        last_blink_time = now
                        last_two_blinks.append(now)
                        if len(last_two_blinks) > 2:
                            last_two_blinks.pop(0)
                        if (len(last_two_blinks) == 2 and
                            (last_two_blinks[1] - last_two_blinks[0]) <= DOUBLE_BLINK_CANCEL_MAX):
                            blink_sequence.clear()
                            predicted_letter = "-"
                        time.sleep(BLINK_COOLDOWN)

            # --- Finalize letter ---
            if last_blink_time and (time.time() - last_blink_time) > PATTERN_TIMEOUT and blink_sequence:
                letter = decode_sequence(blink_sequence)
                final_text += letter
                predicted_letter = "-"
                blink_sequence.clear()
                last_blink_time = None

            # --- Update UI data ---
            status_data.update({
                "ear": float(ear) if ear else 0,
                "pattern": ' '.join(blink_sequence) or "-",
                "preview": predicted_letter or "-",
                "final_text": final_text or "-",
                "speaking": speaking
            })

            # --- Display info on video ---
            cv2.putText(frame, f"EAR: {ear:.2f}" if ear else "EAR: -", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Pattern: {' '.join(blink_sequence) if blink_sequence else '-'}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            cv2.putText(frame, f"Preview: {predicted_letter}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Text: {final_text}", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    Thread(target=run_server, daemon=True).start()
    print("BlinkTalk running at http://127.0.0.1:5000/")
    while True:
        time.sleep(1)
