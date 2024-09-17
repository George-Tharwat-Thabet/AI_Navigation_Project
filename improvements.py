print("Hello AI Navigation")

import requests
import torch
from PIL import Image
import numpy as np
import pyttsx3

# Load the YOLOv5 model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open an image file for object detection
img_path = 'sample.jpg'
img = Image.open(img_path)

# Perform object detection on the image
results = model(img)

# Extract detection data (bounding box coordinates and object names)
detection_data = results.pandas().xyxy[0] 

# Get image dimensions for calculating object positions
img_width, img_height = img.size

# Function to determine where the object is located (left, center, or right) and estimate distance
def get_navigation_info(row, img_width):
    object_name = row['name']
    xmin, xmax = row['xmin'], row['xmax']
    center_x = (xmin + xmax) / 2
    
    if center_x < img_width / 3:
        direction = "left"
    elif center_x > 2 * img_width / 3:
        direction = "right"
    else:
        direction = "center"
    
    ymin = row['ymin']
    distance_estimation = 20 - (ymin / img_height) * 20

    return f"A {object_name} is on the {direction}, approximately {distance_estimation:.1f} meters away."

# List to store navigation descriptions for each detected object
navigation_info = []
for _, row in detection_data.iterrows():
    info = get_navigation_info(row, img_width)
    navigation_info.append(info)

# Combine all the navigation info into one description
surroundings_description = " ".join(navigation_info)

# Print the description of the surroundings
print(surroundings_description)

# Initialize the pyttsx3 engine for TTS (Text-to-Speech)
engine = pyttsx3.init()

# Use TTS to speak out the surroundings description
engine.say(surroundings_description)
engine.runAndWait()

# Improved prompt engineering for the API request
url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
user_input = "How do I get to the nearest chair?"

body = {
    "input": f"""You are an AI assistant designed to help visually impaired individuals navigate their surroundings based on object detection data. Provide clear, concise, and helpful navigation instructions.

Current surroundings:
{surroundings_description}

User question: {user_input}

Instructions:
1. Analyze the surroundings description.
2. Identify the relevant objects for the user's question.
3. Provide a clear and concise response with helpful navigation instructions.
4. If the requested object is not present, suggest the closest alternative or advise the user accordingly.
5. Use cardinal directions (front, back, left, right) and approximate distances for clarity.

Response:""",
    "parameters": {
        "decoding_method": "greedy",
        "max_new_tokens": 150,
        "stop_sequences": ["\n\n"],
        "repetition_penalty": 1.2
    },
    "model_id": "ibm/granite-13b-chat-v2",
    "project_id": "8e0a89d4-f10e-49c9-9f1c-e11c4580adee",
    "moderations": {
        "hap": {
            "input": {
                "enabled": True,
                "threshold": 0.5,
                "mask": {
                    "remove_entity_value": True
                }
            },
            "output": {
                "enabled": True,
                "threshold": 0.5,
                "mask": {
                    "remove_entity_value": True
                }
            }
        }
    }
}

# Access token (replace with a valid token)
accesstoken = " "

# Prepare headers for the API request
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer" + " " + accesstoken
}

# Make the API request
response = requests.post(url, headers=headers, json=body)

# Check for a successful response
if response.status_code != 200:
    raise Exception("Non-200 response: " + str(response.text))

# Parse and print the response
data = response.json()
generated_text = data['results'][0]['generated_text'].strip()
print("AI Assistant:", generated_text)

# Use TTS to speak out the AI assistant's response
engine.say(generated_text)
engine.runAndWait()

# Display the results (image with bounding boxes)
results.show()
