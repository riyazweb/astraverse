import os
import shutil
import base64
import cv2
import numpy as np
import requests
import subprocess
import threading
from time import sleep
from flask import Flask, request, jsonify, render_template
from moviepy.editor import VideoFileClip
from pathlib import Path
from pyngrok import ngrok
import speech_recognition as sr

# Directory constants
UPLOAD_FOLDER = 'recorded_videos'
EXTRACTED_FRAMES = 'extracted_frames'
AUDIO_FOLDER = 'extracted_audio'

# Example model name for your Ollama server
OLLAMA_MODEL = 'llama3.2-vision'  

# Ensure necessary directories exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(EXTRACTED_FRAMES).mkdir(parents=True, exist_ok=True)
Path(AUDIO_FOLDER).mkdir(parents=True, exist_ok=True)

# Optional: Launch an Ollama server in a separate thread
def ollama_server():
    print("Starting Ollama server...")
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'  # Adjust host and port if needed
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])
    print("Ollama server started.")

# Start the Ollama server in a background thread
ollama_thread = threading.Thread(target=ollama_server, daemon=True)
ollama_thread.start()

# Allow some time for the Ollama server to initialize
sleep(5)

app = Flask(__name__)

def clean_folders():
    """
    Clears out all existing files in the upload, frames, and audio folders.
    This ensures a fresh environment for each new video analysis.
    """
    for folder in [UPLOAD_FOLDER, EXTRACTED_FRAMES, AUDIO_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

def extract_audio(video_path, audio_out_path):
    """
    Extracts audio from the given video file using FFmpeg directly.
    This avoids MoviePy’s "failed to read the duration" issue with certain .webm files.
    
    Parameters:
        video_path (str): Path to the input video file.
        audio_out_path (str): Path where the extracted audio will be saved.
    
    Returns:
        bool: True if extraction is successful, False otherwise.
    """
    try:
        # -y: overwrite output, -i: input file, -vn: disable video, -acodec: specify PCM
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",  # PCM format for WAV
            audio_out_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print("Audio extraction error:", e)
        return False

def transcribe_audio(audio_path):
    """
    Transcribes audio using SpeechRecognition's Google Web Speech API.
    
    Parameters:
        audio_path (str): Path to the audio file to transcribe.
    
    Returns:
        str: Transcription result or an error message.
    """
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
    """
    Extracts frames from the video at 1 frame per second.
    
    Parameters:
        video_path (str): Path to the input video file.
    
    Returns:
        list: List of paths to the extracted frame images.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 24  # Fallback to 24 FPS if unable to get FPS
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
    """
    Combines extracted frames into a single image by stacking them vertically.
    
    Parameters:
        frame_paths (list): List of paths to frame images.
        output_file (str): Filename for the combined image.
    
    Returns:
        str or None: Path to the combined image or None if no frames are available.
    """
    if not frame_paths:
        return None

    images = []
    for frame_file in frame_paths:
        img = cv2.imread(frame_file)
        if img is not None:
            height, width = img.shape[:2]
            max_width = 400
            new_height = int((max_width / width) * height)
            resized = cv2.resize(img, (max_width, new_height))
            images.append(resized)

    if not images:
        return None

    combined = np.vstack(images)
    combined_path = os.path.join(EXTRACTED_FRAMES, output_file)
    cv2.imwrite(combined_path, combined)
    return combined_path

def query_ollama(prompt, image_path=None):
    """
    Sends a prompt and optional image to the Ollama server for analysis.
    
    Parameters:
        prompt (str): The text prompt for the AI model.
        image_path (str, optional): Path to an image file to include in the analysis.
    
    Returns:
        str: The AI model's response or an error message.
    """
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
        # Optionally attach an image if available
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
    """
    Renders the main page.
    """
    return render_template('index.html')

@app.route('/save_video', methods=['POST'])
def save_video():
    """
    Handles the video upload, processes it, extracts audio, transcribes, and analyzes frames.
    
    Returns:
        JSON: Contains transcription and AI analysis results or an error message.
    """
    clean_folders()

    video_path = os.path.join(UPLOAD_FOLDER, "upload.webm")
    with open(video_path, 'wb') as f:
        f.write(request.data)

    try:
        # Extract audio from the video
        audio_path = os.path.join(AUDIO_FOLDER, "extracted.wav")
        success = extract_audio(video_path, audio_path)
        if success:
            transcription = transcribe_audio(audio_path)
        else:
            transcription = "No audio could be extracted."

        # Extract frames for analysis
        frames = extract_frames(video_path)
        if not frames:
            gemini_response = "No frames extracted."
        else:
            # Combine frames into one image
            combined_image = create_combined_image(frames)

            # Create a prompt for analysis
            prompt = f"Analyze this video content. Transcription: \"{transcription}\". Summarize the visual content in two sentences."

            # Query Ollama server
            gemini_response = query_ollama(prompt, combined_image)

        return jsonify({
            "gemini_response": gemini_response,
            "transcription": transcription
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # (Optional) Use ngrok to expose the Flask server externally
    # Remove or comment out if not needed
    public_url = ngrok.connect(5000)
    print("Public URL:", public_url)

    # Start the Flask server
    app.run(port=5000, debug=False)
