import torch
import cv2
import numpy as np
import pyttsx3
import requests
from typing import List, Tuple, Dict, Optional

def load_yolov5_model() -> Optional[torch.nn.Module]:
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        return None

def detect_objects(model: torch.nn.Module, image_path: str) -> Tuple[Optional[List[Dict]], int, int]:
    try:
        img = cv2.imread(image_path)
        results = model(img)
        img_height, img_width = img.shape[:2]
        
        detection_data = []
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            label = results.names[int(cls)]
            detection_data.append({
                'label': label,
                'confidence': float(conf),
                'bbox': [x1, y1, x2, y2]
            })
        
        return detection_data, img_width, img_height
    except Exception as e:
        print(f"Error detecting objects: {e}")
        return None, 0, 0

def get_navigation_info(bbox: List[int], img_width: int, img_height: int) -> Tuple[str, str]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    
    if center_x < img_width / 3:
        location = "left"
    elif center_x < 2 * img_width / 3:
        location = "center"
    else:
        location = "right"
    
    object_height = y2 - y1
    distance_ratio = object_height / img_height
    
    if distance_ratio > 0.5:
        distance = "very close"
    elif distance_ratio > 0.3:
        distance = "close"
    elif distance_ratio > 0.1:
        distance = "moderate distance"
    else:
        distance = "far"
    
    return location, distance

def generate_surroundings_description(detection_data: List[Dict], img_width: int, img_height: int) -> str:
    descriptions = []
    for obj in detection_data:
        label = obj['label']
        location, distance = get_navigation_info(obj['bbox'], img_width, img_height)
        descriptions.append(f"A {label} is {distance} to your {location}")
    
    return ". ".join(descriptions) + "."

def speak_surroundings_description(description: str) -> None:
    engine = pyttsx3.init()
    engine.say(description)
    engine.runAndWait()

def generate_response(surroundings_description: str, user_input: str) -> Optional[str]:
    url = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29"
    
    prompt = f"""You are an AI assistant designed to help visually impaired individuals navigate their surroundings. 
    Your task is to provide clear, concise, and helpful responses based on the following information:

    Surroundings description: {surroundings_description}

    User's question: {user_input}

    Please provide a response that:
    1. Directly addresses the user's question
    2. Uses the surroundings description to give context-aware guidance
    3. Prioritizes safety and clarity in your instructions
    4. Avoids assumptions about the user's abilities or the environment beyond what's described

    Response:"""

    body = {
        "model_id": "google/flan-ul2",
        "input": prompt,
        "parameters": {
            "decoding_method": "sample",
            "max_new_tokens": 250,
            "min_new_tokens": 50,
            "random_seed": 111,
            "stop_sequences": [],
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 1,
            "repetition_penalty": 1
        },
        "project_id": "REPLACE_WITH_YOUR_PROJECT_ID"
    }

    try:
        accesstoken = "REPLACE_WITH_YOUR_ACCESS_TOKEN"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {accesstoken}"
        }

        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return data['results'][0]['generated_text'].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def main():
    model = load_yolov5_model()
    if model:
        detection_data, img_width, img_height = detect_objects(model, 'sample.jpg')
        if detection_data is not None:
            surroundings_description = generate_surroundings_description(detection_data, img_width, img_height)
            print(f"Surroundings description: {surroundings_description}")
            speak_surroundings_description(surroundings_description)
            
            user_input = input("What would you like to know about your surroundings? ")
            response = generate_response(surroundings_description, user_input)
            if response:
                print(f"AI Assistant: {response}")
                speak_surroundings_description(response)

if __name__ == "__main__":
    main()

print("Code has been updated with improvements in modularity, error handling, and prompt engineering.")
