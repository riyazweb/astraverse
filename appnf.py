import os
import threading
import subprocess
import requests
import json
import base64
import cv2
import numpy as np
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from pyngrok import ngrok
from time import sleep

# Configuration
UPLOAD_FOLDER = 'recorded_videos'
EXTRACTED_FRAMES = 'extracted_frames'
OLLAMA_MODEL = 'llama3.2-vision'  # Updated vision model name

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(EXTRACTED_FRAMES).mkdir(parents=True, exist_ok=True)

# Start Ollama server
def ollama_server():
    print("🟠 Starting Ollama server...")
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])
    print("🟢 Ollama server started")

ollama_thread = threading.Thread(target=ollama_server)
ollama_thread.start()
sleep(5)  # Give the Ollama server time to initialize

def extract_frames(video_path):
    """Extract one frame per second from video."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(fps, 1)  # At least 1 frame per second
    frame_count = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = str(Path(EXTRACTED_FRAMES) / f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
        
        frame_count += 1

    cap.release()
    return saved_frames

def create_combined_image(frame_paths, output_path="combined.jpg", max_width=400):
    """Combine frames into a single vertical image."""
    if not frame_paths:
        return None

    images = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            new_h = int((max_width / w) * h)
            images.append(cv2.resize(img, (max_width, new_h)))

    if not images:
        return None

    combined = np.vstack(images)
    cv2.imwrite(output_path, combined)
    return output_path

def get_ollama_response(prompt, image_path):
    """Get response from Ollama vision model."""
    url = 'http://localhost:11434/api/generate'
    
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode('utf-8')

    data = {
          "model": OLLAMA_MODEL,
          "prompt": prompt,
          "stream": False,
          "images": [b64_image],
          "options": {
              "temperature": 0.3,
              "top_p": 0.9,
              "num_ctx": 4096
          }
      }
  

    try:
        response = requests.post(url, json=data, timeout=90)
        response.raise_for_status()
        return response.json().get('response', 'No response')
    except Exception as e:
        return f"Error: {str(e)}"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def clean_folders():
    """Clean all working directories before processing."""
    for folder in [UPLOAD_FOLDER, EXTRACTED_FRAMES]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('indexf.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    clean_folders()

    # Save video
    video_path = Path(UPLOAD_FOLDER) / "upload.webm"
    with open(video_path, 'wb') as f:
        f.write(request.data)

    # Process the video
    try:
        frames = extract_frames(str(video_path))
        if not frames:
            return jsonify({"error": "No frames extracted"}), 400

        combined_image = create_combined_image(frames)
        if not combined_image:
            return jsonify({"error": "Image processing failed"}), 500

        # Send frames to Ollama for analysis
        prompt = """Analyze this dynamic visual sequence as temporal moments from continuous motion. 
        Describe the ongoing actions, environmental context, and their progression in two concise 
        natural language sentences. Respond strictly as if observing real-time footage, never 
        mentioning images/frames/visuals. Focus on: 
        1. Movement progression between elements 
        2. Temporal relationships 
        3. Dynamic scene changes 
        4. Continuous narrative flow reply in only 2 lines"""
        analysis = get_ollama_response(prompt, combined_image)

        return jsonify({
            "gemini_response": analysis,
            "image": f"data:image/jpeg;base64,{base64.b64encode(open(combined_image, 'rb').read()).decode('utf-8')}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"🌐 Ngrok URL: {public_url}")
    app.run(debug=False, port=5000)
