import requests
import json

# --- Configuration ---
# 1. This should be your deployed Cloud Run URL
base_url = "https://misinformation-service-737726244243.us-central1.run.app" 

# 2. This is the claim you want to test
claim_to_test = "Drinking lemonade with honey reduces risk of throat cancer"

# 3. The endpoint is at the root "/"
test_url = f"{base_url}/"

# 4. Data is now sent as form data in a POST request
form_data = {
    "text": claim_to_test
}

try:
    print(f"▶️  Sending POST request to: {test_url}")
    print(f"▶️  Claim: \"{claim_to_test}\"")

    # --- Make the POST request ---
    # We use 'data=' for form data, not 'params='
    response = requests.post(test_url, data=form_data)
    
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
