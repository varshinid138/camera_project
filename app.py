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

app = Flask(__name__)

FORMAT = pyaudio.paInt16  # Correct format for audio recording
CHANNELS = 1  # Set to 1 for mono and 2 for stereo
RATE = 44100  # Sample rate in Hertz
CHUNK = 1024  # Buffer size for audio stream

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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
GAZE_THRESHOLD = 0.25
HISTORY_LENGTH = 5
EYE_AR_THRESH = 0.2
CHECK_INTERVAL = 3
PICTURE_THRESHOLD = 5


prev_gray = None
face_motions = {}
last_check_time = time.time()


audio_queue = queue.Queue()
audio_stream = None
fs = 44100  # Sample rate

# Video processing functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_center(landmarks, eye_indices):
    return np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices], axis=0)

def detect_gaze(landmarks):
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES])
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    
    if avg_ear < EYE_AR_THRESH:
        return False, " "
    
    left_eye_center = get_eye_center(landmarks, LEFT_EYE_INDICES)
    right_eye_center = get_eye_center(landmarks, RIGHT_EYE_INDICES)
    eyes_center = np.mean([left_eye_center, right_eye_center], axis=0)
    
    nose_bridge = np.array([landmarks.part(30).x, landmarks.part(30).y])
    face_width = landmarks.part(16).x - landmarks.part(0).x
    
    distance = np.linalg.norm(eyes_center - nose_bridge)
    looking_at_camera = (distance / face_width) < GAZE_THRESHOLD
    
    return looking_at_camera, " " if looking_at_camera else " "

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    return faces

def process_frame(frame):
    global prev_gray, last_check_time, face_motions

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_cv = detect_bounding_box(frame)
    faces_dlib = detector(gray)

    face_areas = [w * h for (x, y, w, h) in faces_cv]
    main_interviewer_idx = face_areas.index(max(face_areas)) if face_areas else None

    current_time = time.time()

    if current_time - last_check_time >= CHECK_INTERVAL:
        last_check_time = current_time

        if prev_gray is not None and len(faces_cv) > 0:
            for i, (x, y, w, h) in enumerate(faces_cv):
                face_id = f"face_{i}"

                face_diff = cv2.absdiff(prev_gray[y:y+h, x:x+w], gray[y:y+h, x:x+w])
                _, face_thresh = cv2.threshold(face_diff, 25, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(face_thresh)

                if motion_pixels > 50:
                    face_motions[face_id] = {"last_motion": current_time, "status": "person detected"}
                elif face_id not in face_motions:
                    face_motions[face_id] = {"last_motion": current_time - PICTURE_THRESHOLD - 1, "status": "picture detected"}

    for face_id, data in face_motions.items():
        if current_time - data["last_motion"] >= PICTURE_THRESHOLD:
            data["status"] = " "
        else:
            data["status"] = " "

    gaze_history = deque(maxlen=HISTORY_LENGTH)
    looking_at_camera = False
    gaze_status = " "

    for face in faces_dlib:
        landmarks = predictor(gray, face)
        looking_at_camera, gaze_status = detect_gaze(landmarks)
        gaze_history.append(looking_at_camera)
        
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    looking_at_camera = sum(gaze_history) > len(gaze_history) // 2

    for i, (x, y, w, h) in enumerate(faces_cv):
        face_id = f"face_{i}"

        if face_id in face_motions:
            status = face_motions[face_id]["status"]
            color = (0, 255, 0) if status == "person detected" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if i == main_interviewer_idx:
            cv2.putText(frame, '', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif face_areas[i] < face_areas[main_interviewer_idx] * 0.6:
            cv2.putText(frame, '', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, gaze_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    prev_gray = gray.copy()
    
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
    global  output_frame, video_lock, video_capture
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame = process_frame(frame)

        with video_lock:
            output_frame = frame.copy()

def save_to_wav(filename):
    
    try:
        # Create a wave file and set its parameters
        with wave.open(filename, 'wb') as wf:
            # Check and set the number of channels (should be 1 or 2 typically)
            if CHANNELS not in [1, 2]:
                raise ValueError(f"Invalid number of channels: {CHANNELS}. Should be 1 (mono) or 2 (stereo).")
            wf.setnchannels(CHANNELS)
            
            # Get the sample width from the PyAudio format
            audio = pyaudio.PyAudio()
            sample_width = audio.get_sample_size(FORMAT)
            
            if sample_width <= 0:
                raise ValueError(f"Invalid format for sample width: {FORMAT}")
            wf.setsampwidth(sample_width)
            
            # Check and set the frame rate (sampling rate)
            if RATE <= 0:
                raise ValueError(f"Invalid sample rate: {RATE}. It should be a positive integer.")
            wf.setframerate(RATE)
            
            # Write the audio frames to the file
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
    
    # Start Video Capture
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)
        threading.Thread(target=process_video).start()

    # Start Audio Capture
    is_recording = True
    threading.Thread(target=audio_recording_thread).start()
    print("Recording started (audio and video)")

def stop_recording():
    global is_recording, video_capture
    
    # Stop Audio Capture
    is_recording = False
    save_to_wav(RECORDING_FILENAME)
    print("Recording stopped (audio and video)")
    
folder1 = 'C:/Users/VARSHINI/OneDrive/Desktop/Camera Project/static/questions/dlq'
folder2 = 'C:/Users/VARSHINI/OneDrive/Desktop/Camera Project/static/questions/mlq'
folder3 = 'C:/Users/VARSHINI/OneDrive/Desktop/Camera Project/static/questions/statq'

# Dictionary to keep track of used files in each folder
used_files = {
    folder1: set(),
    folder2: set(),
    folder3: set()
}

# Initialize current_folder to None
current_folder = None

def get_random_audio(exclude_folder=None):
    # List of all folders
    folders = [folder1, folder2, folder3]
    
    # Exclude the current folder to ensure next selection is from a different folder
    if exclude_folder:
        folders = [folder for folder in folders if folder != exclude_folder]
    
    # Randomly select a new folder
    new_folder = random.choice(folders)
    
    # Get all audio files in the selected folder
    all_files = [f for f in os.listdir(new_folder) if f.endswith(('.mp3', '.wav'))]
    
    # Get the set of used files in this folder
    used = used_files[new_folder]
    
    # Determine available files by excluding used files
    available_files = list(set(all_files) - used)
    
    # If all files have been used, reset the used_files set for this folder
    if not available_files:
        used_files[new_folder] = set()
        available_files = all_files
    
    # Select a random audio file from available files
    selected_file = random.choice(available_files)
    
    # Add the selected file to the used_files set
    used_files[new_folder].add(selected_file)
    
    return new_folder, selected_file

@app.route('/ask_question', methods=['GET'])
def ask_question():
    global current_folder
    
    new_folder, selected_file = get_random_audio(exclude_folder=current_folder)
    
    current_folder = new_folder
    
    audio_file_path = os.path.join(new_folder, selected_file)
    
    # Use threading to avoid blocking the Flask route while playing the audio
    threading.Thread(target=playsound, args=(audio_file_path,)).start()
    
    # Provide the audio file URL to the client
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
