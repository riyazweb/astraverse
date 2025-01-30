import os
import shutil
import base64
import cv2
import numpy as np
import requests
import subprocess
import threading
import speech_recognition as sr
from time import sleep
from flask import Flask, request, jsonify, render_template
from moviepy.editor import VideoFileClip
from pathlib import Path
from pyngrok import ngrok

UPLOAD_FOLDER = 'recorded_videos'
EXTRACTED_FRAMES = 'extracted_frames'
AUDIO_FOLDER = 'extracted_audio'
OLLAMA_MODEL = 'llama3.2-vision'  # Example model name (replace as needed)

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(EXTRACTED_FRAMES).mkdir(parents=True, exist_ok=True)
Path(AUDIO_FOLDER).mkdir(parents=True, exist_ok=True)

# Optional: Start an Ollama server in a separate thread if you use that for analysis
def ollama_server():
    print("Starting Ollama server...")
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])
    print("Ollama server started.")

ollama_thread = threading.Thread(target=ollama_server)
ollama_thread.start()
# Give the server time to initialize if needed
sleep(3)

app = Flask(__name__)

def clean_folders():
    """Remove existing files/folders before each processing."""
    for folder in [UPLOAD_FOLDER, EXTRACTED_FRAMES, AUDIO_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

def extract_audio(video_path, audio_out_path):
    """Extract audio from video using moviepy."""
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_out_path, codec='pcm_s16le')
        return True
    except Exception as e:
        print("Audio extraction error:", e)
        return False

def transcribe_audio(audio_path):
    """Use speech_recognition to transcribe extracted audio."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_content = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_content)
    except sr.UnknownValueError:
        return "Audio not clear for transcription."
    except sr.RequestError as e:
        return f"Speech Recognition request failed: {e}"

def extract_frames(video_path):
    """Extract one frame per second for analysis if needed."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 24
    frame_interval = int(fps)
    saved_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            out_name = f"frame_{frame_count}.jpg"
            out_path = os.path.join(EXTRACTED_FRAMES, out_name)
            cv2.imwrite(out_path, frame)
            saved_frames.append(out_path)
        frame_count += 1
    cap.release()
    return saved_frames

def create_combined_image(frame_paths, output_file="combined.jpg"):
    """Combine frames vertically into one image."""
    if not frame_paths:
        return None
    images = []
    for f in frame_paths:
        img = cv2.imread(f)
        if img is not None:
            height, width = img.shape[:2]
            max_width = 400
            new_height = int((max_width / width) * height)
            resized = cv2.resize(img, (max_width, new_height))
            images.append(resized)
    if not images:
        return None
    combined = np.vstack(images)
    cv2.imwrite(output_file, combined)
    return output_file

def query_ollama(prompt, image_path=None):
    """Send data to your Ollama server."""
    try:
        url = 'http://localhost:11434/api/generate'
        data = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode('utf-8')
            data["images"] = [b64_img]
        resp = requests.post(url, json=data, timeout=90)
        resp.raise_for_status()
        return resp.json().get('response', 'No response from Ollama.')
    except Exception as e:
        return f"Ollama query error: {str(e)}"

@app.route('/')
def index():
    # Assume index.html is placed in templates/ folder
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    clean_folders()

    # Save the incoming webm file
    blob_path = os.path.join(UPLOAD_FOLDER, "upload.webm")
    with open(blob_path, 'wb') as f:
        f.write(request.data)

    try:
        # Extract audio for transcription
        audio_path = os.path.join(AUDIO_FOLDER, "extracted.wav")
        success = extract_audio(blob_path, audio_path)
        if success:
            transcription = transcribe_audio(audio_path)
        else:
            transcription = "No audio extracted."

        # Optionally extract frames for visual analysis
        frames = extract_frames(blob_path)
        if not frames:
            # If no frames, we can still return the transcription
            return jsonify({
                "ollama_response": f"No frames extracted.\nTranscription: {transcription}"
            })

        # Combine frames
        combined_image = create_combined_image(frames, "combined.jpg")

        # Build a prompt that includes the transcription
        prompt = f"""Analyze this video content. 
        Audio transcription: \"{transcription}\" 
        Summarize what you see in the frames above in 2 sentences."""

        # Call Ollama or any other AI endpoint
        analysis_text = query_ollama(prompt, combined_image)

        return jsonify({
            "ollama_response": analysis_text,
            "transcription": transcription
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use ngrok if you want an external URL
    public_url = ngrok.connect(5000)
    print("Public URL:", public_url)
    app.run(port=5000, debug=False)
