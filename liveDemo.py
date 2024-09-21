import cv2
import requests
import torch
from PIL import Image
import numpy as np
import pyttsx3
import speech_recognition as sr
import threading
import time

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

# Function to determine where the object is located and estimate distance
def get_navigation_info(row, img_width, img_height):
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

# Load the YOLOv5 model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the pyttsx3 engine for TTS
engine = pyttsx3.init()

# Start video capture
cap = cv2.VideoCapture(0)

# Reduce frame size for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variable to store surroundings description
surroundings_description = ""

# Function to process frames
def process_frame(frame):
    global surroundings_description

    # Convert the frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection on the image
    results = model(img)

    # Extract detection data
    detection_data = results.pandas().xyxy[0]

    # Get image dimensions
    img_width, img_height = img.size

    # List to store navigation descriptions
    navigation_info = []
    for _, row in detection_data.iterrows():
        info = get_navigation_info(row, img_width, img_height)
        navigation_info.append(info)

    # Combine all the navigation info
    surroundings_description = " ".join(navigation_info)

    # Display the results (image with bounding boxes)
    annotated_frame = results.render()[0]
    cv2.imshow("YOLOv5 Inference", annotated_frame)

# Thread for processing frames
def frame_processing_thread():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        time.sleep(0.1)  # Limit frame rate to reduce lag

# Start the frame processing thread
thread = threading.Thread(target=frame_processing_thread)
thread.start()

while True:
    # Check for user input (non-blocking)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        user_input = recognize_speech()

        # If the user input is successfully recognized, proceed to speak the surroundings description
        if user_input:
            engine.say(surroundings_description)
            engine.runAndWait()

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
