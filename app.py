import os
import random
from winsound import PlaySound
import cv2
import dlib
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
import time
import threading
from flask import Flask, Response, render_template, jsonify, url_for
import sounddevice as sd
import soundfile as sf
import wave
import queue
from playsound import playsound
import pyaudio
import keyboard
import json

app = Flask(__name__)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Audio processing globals
audio_queue = queue.Queue()
audio_stream = None
audio_frames = []
is_recording = False
RECORDING_FILENAME = "output.wav"

# Video processing globals
video_capture = None
output_frame = None
video_lock = threading.Lock()

# Load the face detection and landmark prediction models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for gaze detection
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
EYE_AR_THRESH = 0.2
GAZE_THRESH = 5

# New variables for moving average
gaze_history = []
history_length = 10

total_frames = 0
center_frames = 0
two_person_frames = 0
start_time = None

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_pupil(eye_region, frame):
    if eye_region.size == 0:
        return None
    try:
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        _, thresh_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    except Exception as e:
        print(f"Error in detect_pupil: {e}")
    return None

def detect_gaze(landmarks, frame):
    def get_eye_points(indices):
        return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in indices])

    left_eye = get_eye_points(LEFT_EYE_INDICES)
    right_eye = get_eye_points(RIGHT_EYE_INDICES)

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    if avg_ear < EYE_AR_THRESH:
        return "Eyes Closed", frame

    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)
    center_point = ((left_eye_center + right_eye_center) / 2).astype(int)

    h, w = frame.shape[:2]
    left_eye_region = frame[max(0, left_eye_center[1]-10):min(h, left_eye_center[1]+10), 
                            max(0, left_eye_center[0]-15):min(w, left_eye_center[0]+15)]
    right_eye_region = frame[max(0, right_eye_center[1]-10):min(h, right_eye_center[1]+10), 
                             max(0, right_eye_center[0]-15):min(w, right_eye_center[0]+15)]

    left_pupil = detect_pupil(left_eye_region, frame)
    right_pupil = detect_pupil(right_eye_region, frame)

    if left_pupil is None or right_pupil is None:
        return "Pupil not detected", frame

    left_pupil_pos = (left_eye_center[0] - 15 + left_pupil[0], left_eye_center[1] - 10 + left_pupil[1])
    right_pupil_pos = (right_eye_center[0] - 15 + right_pupil[0], right_eye_center[1] - 10 + right_pupil[1])

    left_distance = center_point[0] - left_pupil_pos[0]
    right_distance = right_pupil_pos[0] - center_point[0]

    cv2.circle(frame, tuple(center_point), 3, (255, 0, 0), -1)
    cv2.circle(frame, left_pupil_pos, 3, (0, 255, 0), -1)
    cv2.circle(frame, right_pupil_pos, 3, (0, 255, 0), -1)

    if abs(left_distance - right_distance) < GAZE_THRESH:
        return "Center", frame
    elif left_distance > right_distance:
        return "Right", frame
    else:
        return "Left", frame

def process_frame(frame):
    global total_frames, center_frames, two_person_frames, gaze_history

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 2:
        two_person_frames += 1
        cv2.putText(frame, "Two Persons Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    for face in faces:
        landmarks = predictor(gray, face)
        try:
            gaze_direction, frame = detect_gaze(landmarks, frame)
            
            gaze_history.append(gaze_direction)
            if len(gaze_history) > history_length:
                gaze_history.pop(0)
            
            smoothed_gaze = max(set(gaze_history), key=gaze_history.count)
            
            if smoothed_gaze == "Center":
                center_frames += 1
                cv2.putText(frame, "Center", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, smoothed_gaze, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error in gaze detection: {e}")
    
    total_frames += 1
    
    return frame

def generate():
    global output_frame, video_lock

    while True:
        with video_lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

def process_video():
    global output_frame, video_lock, video_capture, start_time
    
    start_time = time.time()
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame = process_frame(frame)

        with video_lock:
            output_frame = frame.copy()

        if time.time() - start_time >= 120:  # Run for 120 seconds
            break

    # After video processing, save results
    save_results()

def save_results():
    global total_frames, center_frames, two_person_frames

    percentage_looking_at_center = (center_frames / total_frames) * 100 if total_frames else 0
    percentage_two_persons = (two_person_frames / total_frames) * 100 if total_frames else 0

    results = {
        "total_frames": total_frames,
        "frames_looking_at_center": center_frames,
        "frames_with_two_persons": two_person_frames,
        "percentage_looking_at_center": round(percentage_looking_at_center, 2),
        "percentage_frames_with_two_persons": round(percentage_two_persons, 2),
        "success": percentage_looking_at_center >= 70
    }

    with open('gaze_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Analysis complete. Results have been saved to gaze_analysis_results.json")

# Rest of the code remains the same...

def save_to_wav(filename):
    try:
        with wave.open(filename, 'wb') as wf:
            if CHANNELS not in [1, 2]:
                raise ValueError(f"Invalid number of channels: {CHANNELS}. Should be 1 (mono) or 2 (stereo).")
            wf.setnchannels(CHANNELS)
            
            audio = pyaudio.PyAudio()
            sample_width = audio.get_sample_size(FORMAT)
            
            if sample_width <= 0:
                raise ValueError(f"Invalid format for sample width: {FORMAT}")
            wf.setsampwidth(sample_width)
            
            if RATE <= 0:
                raise ValueError(f"Invalid sample rate: {RATE}. It should be a positive integer.")
            wf.setframerate(RATE)
            
            wf.writeframes(b''.join(audio_frames))
    except Exception as e:
        print(f"Error saving WAV file: {e}")

def audio_recording_thread():
    global is_recording, audio_frames
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    audio_frames.clear()

    while is_recording:
        data = stream.read(CHUNK)
        audio_frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

def start_recording():
    global is_recording, video_capture
    
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)
        threading.Thread(target=process_video).start()

    is_recording = True
    threading.Thread(target=audio_recording_thread).start()
    print("Recording started (audio and video)")

def stop_recording():
    global is_recording, video_capture
    
    is_recording = False
    save_to_wav(RECORDING_FILENAME)
    print("Recording stopped (audio and video)")
    
folder1 = 'C:/Users/VARSHINI/OneDrive/Desktop/Camera Project/static/questions/dlq'
folder2 = 'C:/Users/VARSHINI/OneDrive/Desktop/Camera Project/static/questions/mlq'
folder3 = 'C:/Users/VARSHINI/OneDrive/Desktop/Camera Project/static/questions/statq'

used_files = {
    folder1: set(),
    folder2: set(),
    folder3: set()
}

current_folder = None

def get_random_audio(exclude_folder=None):
    folders = [folder1, folder2, folder3]
    
    if exclude_folder:
        folders = [folder for folder in folders if folder != exclude_folder]
    
    new_folder = random.choice(folders)
    
    all_files = [f for f in os.listdir(new_folder) if f.endswith(('.mp3', '.wav'))]
    
    used = used_files[new_folder]
    
    available_files = list(set(all_files) - used)
    
    if not available_files:
        used_files[new_folder] = set()
        available_files = all_files
    
    selected_file = random.choice(available_files)
    
    used_files[new_folder].add(selected_file)
    
    return new_folder, selected_file

@app.route('/ask_question', methods=['GET'])
def ask_question():
    global current_folder
    
    new_folder, selected_file = get_random_audio(exclude_folder=current_folder)
    
    current_folder = new_folder
    
    audio_file_path = os.path.join(new_folder, selected_file)
    
    threading.Thread(target=playsound, args=(audio_file_path,)).start()
    
    question_audio_url = url_for('static', filename=os.path.join('questions', new_folder.split('/')[-1], selected_file))
    
    return jsonify({"message": "Playing question", "question_audio_url": question_audio_url})

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/start')
def start_recording_route():
    start_recording()
    return jsonify({'message': 'Recording started'}), 200

@app.route('/stop')
def stop_recording_route():
    stop_recording()
    return jsonify({'message': 'Recording stopped'}), 200

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    t = threading.Thread(target=process_video)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port="5000", debug=True,
        threaded=True, use_reloader=False)

    video_capture.release()


    