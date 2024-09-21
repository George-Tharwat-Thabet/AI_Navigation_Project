import requests
import torch
from PIL import Image
import numpy as np
import speech_recognition as sr  # Importing SpeechRecognition for speech-to-text conversion

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

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Small version of YOLOv5

img_path = 'sample.jpg'
img = Image.open(img_path)

results = model(img)

detection_data = results.pandas().xyxy[0] 

img_width, img_height = img.size

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

navigation_info = []
for _, row in detection_data.iterrows():
    info = get_navigation_info(row, img_width)
    navigation_info.append(info)

surroundings_description = " ".join(navigation_info)

print(surroundings_description)

url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

# Speech-to-text for user input (real-time command processing)
user_input = "How the surroundings"
print(user_input)

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
{user_input}
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

accesstoken = ""

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer"+" "+accesstoken
}

response = requests.post(url, headers=headers, json=body)

if response.status_code != 200:
    raise Exception("Non-200 response: " + str(response.text))

data = response.json()

print(data)

results.show()
