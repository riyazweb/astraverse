from flask import Flask, render_template, request, jsonify
import os
import shutil
import google.generativeai as genai
import time
import dotenv

# Load environment variables from a .env file
dotenv.load_dotenv()

# Set up your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Function to clear the upload folder
def clear_upload_folder():
    folder = "recorded_videos"
    if not os.path.exists(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Clear the upload folder at the start of the application
clear_upload_folder()

app = Flask(__name__)

# Folder to store recorded videos
app.config['UPLOAD_FOLDER'] = 'recorded_videos'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_video', methods=['POST'])
def save_video():
    """
    Saves the uploaded video, uploads it to Gemini, processes it,
    and returns the Gemini response.
    """
    # Read the raw binary data from the HTTP request body
    video_data = request.data

    # Save the video to recorded_videos/rec.webm
    file_name = "rec.webm"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Write the video data to a file
    with open(file_path, 'wb') as f:
        f.write(video_data)

    # Upload the video to the Gemini API
    try:
        print("Uploading file to Gemini...")
        video_file = genai.upload_file(path=file_path)
        print(f"Completed upload: {video_file.uri}")

        # Wait for the file to be processed
        while video_file.state.name == "PROCESSING":
            print("Waiting for video to be processed...")
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed.")
        print(f"Video processing complete: {video_file.uri}")

        # Generate content using the uploaded video
        prompt = "Describe this video."
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

        print("Making LLM inference request to Gemini...")
        response = model.generate_content([prompt, video_file], request_options={"timeout": 600})
        print("Received response from Gemini.")

        gemini_response_text = response.text

        # Delete the uploaded file manually (optional)
        genai.delete_file(video_file.name)
        print(f"Deleted file {video_file.uri}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({
            "message": "Video processing failed.",
            "error": str(e)
        }), 500

    return jsonify({
        "message": "Video saved and processed successfully.",
        "file_path": file_path,
        "gemini_response": gemini_response_text
    })

if __name__ == '__main__':
    app.run(debug=True)