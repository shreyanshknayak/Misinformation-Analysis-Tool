import requests
import sys
import json

def test_image_timeline_endpoint(image_path: str):
    """
    Sends a local image to the timeline endpoint and prints the result.
    """
    
    # This is the specific URL you provided
    endpoint_url = "https://image-timeline-api-737726244243.us-central1.run.app/generate-timeline"
    
    print(f"🚀 Sending '{image_path}' to endpoint...")
    print(f"   URL: {endpoint_url}")

    try:
        # Open the local image file in binary read mode ('rb')
        with open(image_path, 'rb') as f:
            
            # The 'files' dictionary maps the form field name ('file')
            # to the file object. This must match the 'file' argument
            # in your FastAPI endpoint.
            files = {'file': f}

            # Make the POST request
            # This request might take a while as it's doing:
            # 1. GCS Upload
            # 2. Vision API call
            # 3. Web Scraping
            # 4. Vertex AI call
            response = requests.post(endpoint_url, files=files, timeout=120) # 2 min timeout

            # Check for HTTP errors (like 4xx or 5xx)
            response.raise_for_status()

            # If successful, print the JSON response
            print(f"\n✅ Success! (Status Code: {response.status_code})")
            print("--- Response ---")
            
            # Pretty-print the JSON response
            try:
                print(json.dumps(response.json(), indent=2))
            except requests.exceptions.JSONDecodeError:
                print("Error: Received a non-JSON response from the server.")
                print(response.text)


    except FileNotFoundError:
        print(f"❌ ERROR: Image file not found at '{image_path}'")
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Could not connect to the endpoint.")
        print("       Is the URL correct and the Cloud Run service running?")
    except requests.exceptions.ReadTimeout:
        print(f"❌ ERROR: The request timed out.")
        print("       The server is taking too long to process the image.")
    except requests.exceptions.HTTPError as e:
        print(f"❌ ERROR: HTTP Error {e.response.status_code} - {e.response.reason}")
        print(f"       Server detail: {e.response.text}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_endpoint.py <IMAGE_PATH>")
        print("Example:")
        print("  python test_endpoint.py ./my_test_image.jpg")
        sys.exit(1)
        
    image_path = sys.argv[1]
    test_image_timeline_endpoint(image_path)