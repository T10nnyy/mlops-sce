import requests
import json

# The URL of the Flask app running locally
url = 'http://127.0.0.1:5000/predict'

# Example JSON data to be sent to the Flask API
data = {
    "State": "Gujarat",
    "District": "Amreli",
    "Market": "Damnagar",
    "Commodity": "Bhindi(Ladies Finger)",
    "Variety": "Bhindi",
    "Grade": "FAQ",
    "Min Price": 4100,
    "Max Price": 4500,
    "Day": 27,
    "Month": 7,
    "Year": 2023
}

# Send the POST request
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    try:
        # Print the JSON response from the server
        print(response.json())  # This should give you the predicted modal price
    except json.JSONDecodeError:
        print("Error: The response could not be parsed as JSON.")
else:
    print(f"Error: Received status code {response.status_code}")
    print("Response Text:", response.text)  # Check if there's an error message in the response
