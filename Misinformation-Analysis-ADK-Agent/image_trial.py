import requests
import json
import os
import mimetypes

# --- Configuration ---
# 1. This should be your deployed Cloud Run URL
base_url = "https://misinformation-service-737726244243.us-central1.run.app" 

# 2. IMPORTANT: Replace this with the actual path to an image or video
# e.g., "my_test_image.jpg" or "my_video.mp4"
file_path = "pak_fake.png" 

# 3. You can optionally send text along with the file
accompanying_text = "What is this image about? Is this a real event?"

# 4. The endpoint is at the root "/"
test_url = f"{base_url}/"
# ---

# Check if the file exists
if not os.path.exists(file_path):
    print(f"❌ Error: File not found at '{file_path}'")
    print("Please update the 'file_path' variable in this script.")
    exit()

# This dictionary will hold the text form data
form_data = {
    "text": accompanying_text
}

try:
    print(f"▶️  Sending POST request with file to: {test_url}")
    print(f"▶️  File: \"{file_path}\"")

    # Open the file in binary-read mode
    with open(file_path, "rb") as file_handle:
        
        # Guess the MIME type (e.g., 'image/jpeg', 'video/mp4')
        file_name = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'application/octet-stream' # Fallback
        
        # This dictionary will hold the file(s) to upload
        # The key "file" must match the parameter name in your FastAPI endpoint
        files_to_upload = {
            "file": (file_name, file_handle, mime_type)
        }
        
        # Make the POST request
        # requests will automatically create a multipart/form-data POST
        response = requests.post(test_url, data=form_data, files=files_to_upload)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("\n✅ Success! Agent returned a response:")
        print("---")
        # Parse the JSON response and print it nicely
        print(json.dumps(response.json(), indent=2))
        print("---")
        
    else:
        # Show error details if something went wrong
        print(f"\n❌ Error: Request failed with status code {response.status_code}")
        print(f"Details: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"\n❌ An error occurred with the network request: {e}")
    print("Please make sure your service is deployed and running.")

