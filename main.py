print("Hello AI Navigation")

import requests  # Importing requests to handle HTTP requests
import torch  # Importing torch for deep learning and tensor operations
from PIL import Image  # Importing Image from PIL for image processing
import numpy as np  # Importing numpy for numerical operations and array handling
import pyttsx3  # Importing pyttsx3 for Text-to-Speech (TTS)
import speech_recognition as sr  # Importing SpeechRecognition for speech-to-text conversion

# Initialize the speech recognition engine
recognizer = sr.Recognizer()

# Function to convert speech to text
def recognize_speech():
    with sr.Microphone() as source:
        print("Listening for input...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            print("Recognizing speech...")
            user_input = recognizer.recognize_google(audio)
            print(f"User said: {user_input}")
            return user_input
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, the speech service is unavailable.")
            return None

# Load the YOLOv5 model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Small version of YOLOv5

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
    object_name = row['name']  # Get the name of the detected object
    xmin, xmax = row['xmin'], row['xmax']
    center_x = (xmin + xmax) / 2  # Calculate the center position of the object
    
    # Determine if the object is on the left, right, or center
    if center_x < img_width / 3:
        direction = "left"
    elif center_x > 2 * img_width / 3:
        direction = "right"
    else:
        direction = "center"
    
    # Estimate the distance of the object based on its vertical position in the image
    ymin = row['ymin']
    distance_estimation = 20 - (ymin / img_height) * 20  # A simple estimation of distance in meters

    return f"A {object_name} is on the {direction}, approximately {distance_estimation:.1f} meters away."

# List to store navigation descriptions for each detected object
navigation_info = []
for _, row in detection_data.iterrows():
    info = get_navigation_info(row, img_width)  # Get the navigation info for each detected object
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

# Speech-to-text for user input (real-time command processing)
user_input = recognize_speech()

# If the user input is successfully recognized, proceed to API request
if user_input:
    # Code to send the surroundings description and user input to an external API for text generation
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    
    body = {
	"input": f"""Navigate the person about the surroundings.

	Input: Hello, can you help me with directions?
	Output: \"Hello! I'\''d be happy to help. What do you need directions to?\"

	Input: A laptop is on the left, approximately 17.6 meters away.
	A chair is in the center, approximately 18.5 meters away.
	A book is on the right, approximately 9.4 meters away.
	A bottle is on the right, approximately 17.6 meters away.
	A chair is on the right, approximately 19.2 meters away.
	A chair is on the left, approximately 20.0 meters away.
	A chair is on the right, approximately 19.7 meters away.
	A chair is on the right, approximately 19.8 meters away.
	A chair is on the right, approximately 17.4 meters away.
	A laptop is on the right, approximately 8.4 meters away.
	A book is on the right, approximately 8.5 meters away.
	How do I get to the nearest chair?
	Output: \"There are several chairs around you. To your right, you can find a chair in the center and others further away. There is also a chair on your left side.\"

	Input: \"A laptop is on the left, approximately 17.6 meters away.
	A chair is in the center, approximately 18.5 meters away.
	A book is on the right, approximately 9.4 meters away.
	A bottle is on the right, approximately 17.6 meters away.
	A chair is on the right, approximately 19.2 meters away.
	A chair is on the left, approximately 20.0 meters away.
	A chair is on the right, approximately 19.7 meters away.
	A chair is on the right, approximately 19.8 meters away.
	A chair is on the right, approximately 17.4 meters away.
	A laptop is on the right, approximately 8.4 meters away.
	A book is on the right, approximately 8.5 meters away.
	Where is the nearest book?\"
	Output: \"There are books on both sides. One book is on your right side, relatively close, and another book is a bit further away on the same side.\"

	Input: \"A laptop is on the left, approximately 17.6 meters away.
	A chair is in the center, approximately 18.5 meters away.
	A book is on the right, approximately 9.4 meters away.
	A bottle is on the right, approximately 17.6 meters away.
	A chair is on the right, approximately 19.2 meters away.
	A chair is on the left, approximately 20.0 meters away.
	A chair is on the right, approximately 19.7 meters away.
	A chair is on the right, approximately 19.8 meters away.
	A chair is on the right, approximately 17.4 meters away.
	A laptop is on the right, approximately 8.4 meters away.
	A book is on the right, approximately 8.5 meters away.
	Where is the nearest restroom?\"
	Output: I don’t have information about a restroom in the current surroundings. You might need to ask someone nearby or move to a different area.

	Input: \"A laptop is on the left, approximately 17.6 meters away.
	A chair is in the center, approximately 18.5 meters away.
	A book is on the right, approximately 9.4 meters away.
	A bottle is on the right, approximately 17.6 meters away.
	A chair is on the right, approximately 19.2 meters away.
	A chair is on the left, approximately 20.0 meters away.
	A chair is on the right, approximately 19.7 meters away.
	A chair is on the right, approximately 19.8 meters away.
	A chair is on the right, approximately 17.4 meters away.
	A laptop is on the right, approximately 8.4 meters away.
	A book is on the right, approximately 8.5 meters away.
	Can you describe the items around me?\"
	Output: \"To your right, you have a book and a bottle, with several chairs positioned at varying distances. To your left, there are a couple of chairs and a laptop. In the center, there is a chair.\"

	Input:  A laptop is on the left, approximately 17.6 meters away.
	A chair is in the center, approximately 18.5 meters away.
	A book is on the right, approximately 9.4 meters away.
	A bottle is on the right, approximately 17.6 meters away.
	A chair is on the right, approximately 19.2 meters away.
	A chair is on the left, approximately 20.0 meters away.
	A chair is on the right, approximately 19.7 meters away.
	A chair is on the right, approximately 19.8 meters away.
	A chair is on the right, approximately 17.4 meters away.
	A laptop is on the right, approximately 8.4 meters away.
	A book is on the right, approximately 8.5 meters away.
	What’s the arrangement of the objects around me?
	Output: You have several items to your right, including books and chairs. On the left, there are some chairs and a laptop. The arrangement is spread out, with objects positioned at different relative distances.\"

	Input: A TV is in the center, approximately 11.9 meters away.
	A couch is on the left, approximately 8.9 meters away.
	A bed is on the right, approximately 6.7 meters away.
	A bed is on the right, approximately 4.6 meters away.
	A couch is on the right, approximately 4.3 meters away.
	How do I get to the nearest chair?
	Output: \"I'\''m sorry, there is no chair in the current surroundings, but there is a couch to your right, which is the closest seating option.\"


	Input: A TV is in the center, approximately 11.9 meters away.
	A couch is on the left, approximately 8.9 meters away.
	A bed is on the right, approximately 6.7 meters away.
	A bed is on the right, approximately 4.6 meters away.
	A couch is on the right, approximately 4.3 meters away.
	How do I get to the nearest tv?
	Output: The TV is in the center. You can simply walk towards it to view it.

	Input: A elephant is on the center, approximately 13.3 meters away.
	A bird is on the center, approximately 15.9 meters away.
	A zebra is on the center, approximately 6.8 meters away.
	A sheep is on the center, approximately 8.4 meters away.
	A sheep is on the left, approximately 5.2 meters away.
	A giraffe is on the center, approximately 13.1 meters away.
	A sheep is on the center, approximately 7.0 meters away.
	A zebra is on the right, approximately 8.5 meters away.
	How do I get to the  bed?
	Output: I don’t have information about a bed in the current surroundings. You might need to ask someone nearby or move to a different area.

	Input: {surroundings_description}
	What do you want to know about? {user_input}
	Output:""",
		"parameters": {
			"decoding_method": "greedy",
			"max_new_tokens": 300,
			"stop_sequences": ["\n\n"],
			"repetition_penalty": 1
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
    print(data)

# Display the results (image with bounding boxes)
results.show()
