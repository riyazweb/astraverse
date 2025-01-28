
# Import required library
import google.generativeai as genai
import time
import dotenv
import os

# Load environment variables from a .env file
dotenv.load_dotenv()

# Set up your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


# Specify the video file name
video_file_name = r"recorded_videos/rec.webm"

# Upload the video to the Gemini API
print(f"Uploading file...")
video_file = genai.upload_file(path=video_file_name)
print(f"Completed upload: {video_file.uri}")

# Wait for the file to be processed
while video_file.state.name == "PROCESSING":
    print('Waiting for video to be processed...')
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
    raise ValueError("Video processing failed.")
print(f"Video processing complete: {video_file.uri}")

# Generate content using the uploaded video
prompt = "Describe this video.:"
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

print("Making LLM inference request...")
response = model.generate_content([prompt, video_file], request_options={"timeout": 600})
print(response.text)

# Delete the uploaded file manually (optional)
genai.delete_file(video_file.name)
print(f"Deleted file {video_file.uri}")