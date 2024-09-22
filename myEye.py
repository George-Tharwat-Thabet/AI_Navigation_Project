import requests  # Importing requests to handle HTTP requests
import torch  # Importing torch for deep learning and tensor operations
from PIL import Image  # Importing Image from PIL for image processing
import numpy as np  # Importing numpy for numerical operations and array handling
import pyttsx3  # Importing pyttsx3 for Text-to-Speech (TTS)
import speech_recognition as sr  # Importing SpeechRecognition for speech-to-text conversion
import cv2  # Importing OpenCV for camera access

print("Hello AI Navigation")

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

# Capture an image from the camera
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    ret, frame = cap.read()  # Capture a single frame
    cap.release()  # Release the camera

    if not ret:
        print("Error: Could not read frame.")
        return None

    # Convert the captured frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

# Load the YOLOv5 model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Small version of YOLOv5

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

    return f"A {object_name} is on the {direction},."

# Initialize the pyttsx3 engine for TTS (Text-to-Speech)
engine = pyttsx3.init()

# Main loop to wait for user command
while True:
    # Speech-to-text for user input (real-time command processing)
    user_input = recognize_speech()

    if user_input:
        if "recognize objects" in user_input.lower() or "describe" in user_input.lower() or "see" in user_input.lower():
            # Capture an image from the camera for object detection
            img = capture_image_from_camera()
            if img is None:
                raise Exception("Failed to capture image from camera.")

            # Perform object detection on the image
            results = model(img)

            # Extract detection data (bounding box coordinates and object names)
            detection_data = results.pandas().xyxy[0]

            # Get image dimensions for calculating object positions
            img_width, img_height = img.size

            # List to store navigation descriptions for each detected object
            navigation_info = []
            for _, row in detection_data.iterrows():
                info = get_navigation_info(row, img_width)  # Get the navigation info for each detected object
                navigation_info.append(info)

            # Combine all the navigation info into one description
            surroundings_description = " ".join(navigation_info)

            # Print the description of the surroundings
            print(surroundings_description)

            # Use TTS to speak out the surroundings description
            engine.say(surroundings_description)
            engine.runAndWait()

        else:
            # Ensure surroundings_description is defined
            surroundings_description = "No objects recognized yet."

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
            accesstoken = "eyJraWQiOiIyMDI0MDkwMjA4NDIiLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJJQk1pZC02OTIwMDBJWEZRIiwiaWQiOiJJQk1pZC02OTIwMDBJWEZRIiwicmVhbG1pZCI6IklCTWlkIiwianRpIjoiMGE4ZTg3ODQtOTk2Mi00MGE3LTg5ZTktNTBlMTJlYWMyMTRjIiwiaWRlbnRpZmllciI6IjY5MjAwMElYRlEiLCJnaXZlbl9uYW1lIjoiRGhydXYiLCJmYW1pbHlfbmFtZSI6IkFnYXJ3YWwiLCJuYW1lIjoiRGhydXYgQWdhcndhbCIsImVtYWlsIjoiYS5kaHJ1dkBpaXRnLmFjLmluIiwic3ViIjoiYS5kaHJ1dkBpaXRnLmFjLmluIiwiYXV0aG4iOnsic3ViIjoiYS5kaHJ1dkBpaXRnLmFjLmluIiwiaWFtX2lkIjoiSUJNaWQtNjkyMDAwSVhGUSIsIm5hbWUiOiJEaHJ1diBBZ2Fyd2FsIiwiZ2l2ZW5fbmFtZSI6IkRocnV2IiwiZmFtaWx5X25hbWUiOiJBZ2Fyd2FsIiwiZW1haWwiOiJhLmRocnV2QGlpdGcuYWMuaW4ifSwiYWNjb3VudCI6eyJ2YWxpZCI6dHJ1ZSwiYnNzIjoiOGY4MDZmNmVjYTgzNGQ4ZGEwODRlMzZhYmY5ZmYyZGQiLCJpbXNfdXNlcl9pZCI6IjEyNjg5ODgzIiwiZnJvemVuIjp0cnVlLCJpbXMiOiIyNzUwMzYwIn0sImlhdCI6MTcyNzAxODYxNywiZXhwIjoxNzI3MDIyMjE3LCJpc3MiOiJodHRwczovL2lhbS5jbG91ZC5pYm0uY29tL2lkZW50aXR5IiwiZ3JhbnRfdHlwZSI6InVybjppYm06cGFyYW1zOm9hdXRoOmdyYW50LXR5cGU6YXBpa2V5Iiwic2NvcGUiOiJpYm0gb3BlbmlkIiwiY2xpZW50X2lkIjoiZGVmYXVsdCIsImFjciI6MSwiYW1yIjpbInB3ZCJdfQ.dW0wdcgsfn5HyI1b1Eqtk6fTji4FJtkqMGYOwYEpxI-ZiVtSKFOa0-CtnMcZ26quupOebLDjMBE6If3vLzhkOxn9ViYMTcs7DjsgSdFJ4RwzWPd7hj06QQ9-U7hYUAXzKNOMBF-hG6E87IqflIP24Twlrpw1sohcTNRaU1Cq_ghcXxp2zy4rBl3D9rNhAJM5frGm4j46ofWFGBf5qFUMZoDVskiyfWYMT1O7avQuzuIYKYLXHl5_s-LCuLMTTetzt3h_TQZ7MynP4skpvdH0urcLyuJrX_MGcVoZazNwfeakemuiS2h0S1xHLLk9kX-41G7Z9wj_QoKCS_oW4E2dXA"

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

            # Use TTS to speak out the AI response
            ai_response = data['results'][0]['generated_text']
            engine.say(ai_response)
            engine.runAndWait()
