import cv2  # Importing OpenCV for video capture
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

# Function to determine where the object is located (left, center, or right) and estimate distance
def get_navigation_info(row, img_width, img_height):
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

# Load the YOLOv5 model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Small version of YOLOv5

# Initialize the pyttsx3 engine for TTS (Text-to-Speech)
engine = pyttsx3.init()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection on the image
    results = model(img)

    # Extract detection data (bounding box coordinates and object names)
    detection_data = results.pandas().xyxy[0]

    # Get image dimensions for calculating object positions
    img_width, img_height = img.size

    # List to store navigation descriptions for each detected object
    navigation_info = []
    for _, row in detection_data.iterrows():
        info = get_navigation_info(row, img_width, img_height)  # Get the navigation info for each detected object
        navigation_info.append(info)

    # Combine all the navigation info into one description
    surroundings_description = " ".join(navigation_info)

    # Print the description of the surroundings
    print(surroundings_description)

    # Use TTS to speak out the surroundings description
    engine.say(surroundings_description)
    engine.runAndWait()

    # Display the results (image with bounding boxes)
    results.show()

    # Check for user input
    user_input = recognize_speech()

    # If the user input is successfully recognized, proceed to API request
    if user_input:
        # Code to send the surroundings description and user input to an external API for text generation
        url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

        body = {
            "input": f"""Instructions:
1. Analyze the surroundings description.
2. Identify the relevant objects for the user's question.
3. Provide a clear and concise response with helpful navigation instructions.
4. If the requested object is not present, suggest the closest alternative or advise the user accordingly.
5. Use cardinal directions (front, back, left, right) and approximate distances for clarity.

Surroundings: {surroundings_description}
User question: {user_input}
Response:""",
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
        accesstoken = "eyJraWQiOiIyMDI0MDkwMjA4NDIiLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJJQk1pZC02OTIwMDBJWEZRIiwiaWQiOiJJQk1pZC02OTIwMDBJWEZRIiwicmVhbG1pZCI6IklCTWlkIiwianRpIjoiODc1YzJiY2UtOWZlZC00NDBhLWFlZGQtMjhjZjI2NTFjYzU2IiwiaWRlbnRpZmllciI6IjY5MjAwMElYRlEiLCJnaXZlbl9uYW1lIjoiRGhydXYiLCJmYW1pbHlfbmFtZSI6IkFnYXJ3YWwiLCJuYW1lIjoiRGhydXYgQWdhcndhbCIsImVtYWlsIjoiYS5kaHJ1dkBpaXRnLmFjLmluIiwic3ViIjoiYS5kaHJ1dkBpaXRnLmFjLmluIiwiYXV0aG4iOnsic3ViIjoiYS5kaHJ1dkBpaXRnLmFjLmluIiwiaWFtX2lkIjoiSUJNaWQtNjkyMDAwSVhGUSIsIm5hbWUiOiJEaHJ1diBBZ2Fyd2FsIiwiZ2l2ZW5fbmFtZSI6IkRocnV2IiwiZmFtaWx5X25hbWUiOiJBZ2Fyd2FsIiwiZW1haWwiOiJhLmRocnV2QGlpdGcuYWMuaW4ifSwiYWNjb3VudCI6eyJ2YWxpZCI6dHJ1ZSwiYnNzIjoiOGY4MDZmNmVjYTgzNGQ4ZGEwODRlMzZhYmY5ZmYyZGQiLCJpbXNfdXNlcl9pZCI6IjEyNjg5ODgzIiwiZnJvemVuIjp0cnVlLCJpbXMiOiIyNzUwMzYwIn0sImlhdCI6MTcyNjkxNzA5MSwiZXhwIjoxNzI2OTIwNjkxLCJpc3MiOiJodHRwczovL2lhbS5jbG91ZC5pYm0uY29tL2lkZW50aXR5IiwiZ3JhbnRfdHlwZSI6InVybjppYm06cGFyYW1zOm9hdXRoOmdyYW50LXR5cGU6YXBpa2V5Iiwic2NvcGUiOiJpYm0gb3BlbmlkIiwiY2xpZW50X2lkIjoiZGVmYXVsdCIsImFjciI6MSwiYW1yIjpbInB3ZCJdfQ.pvqMQ5cWtPK1lGzCsovOsSlvuDBuXKhiKMQarmR93HchQWkonUzQMlxK45Nf5dw1ydDWXRY_q2dAWLeJPpSwcyh0yKFMZ_Rfeq9DxNG5dQDZJvbcqYQCaT7eyw2XQRZpSpODNezye97CQTgiwHoDZP1YKLtS-LKtjkjSKgFSS4DFV7UQow6NCOd78C4SzLxn5pC3OtCB9aKUHfjt2csenevegjpKsv6M-vFwRKKGtLsc60zmnMIZYfczuOCbAxK0zkRQ-3CnIyK0cqMFqnVTBWBI4JyoYqkpF5l3dgd1J9LB0AVD_fNEWK2CVHZ-hsIaRScQQJ3BhL1NqR6ryLhbSg"

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

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
