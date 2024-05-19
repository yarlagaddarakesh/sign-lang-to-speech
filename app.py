from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import pickle
import pygame
from gtts import gTTS
from googletrans import Translator
import mediapipe as mp
import datetime
import requests
import csv
from difflib import get_close_matches


# Load suggestions from words.csv
def load_suggestions():
    suggestions = {}
    with open('words.csv', 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Check if the row is not empty
                word = row[0]
                first_letter = word[0].upper()
                if first_letter in suggestions:
                    if word not in suggestions[first_letter]:
                        suggestions[first_letter].append(word)
                else:
                    suggestions[first_letter] = [word]
    return suggestions

suggestion_words = load_suggestions()
# print(suggestion_words)

def find_similar_words(user_input, suggestions):
    similar_words = []
    first_letter = user_input[0].upper()
    if first_letter in suggestions:
        similar_words = get_close_matches(user_input, suggestions[first_letter], n=3)
    return similar_words


app = Flask(__name__)

translator = Translator()
pygame.mixer.init()
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'home', 37: 'brother', 38: 'pay', 39: 'justice', 40: 'science',
    41: 'Please', 42: 'Hello', 43: 'You are Welcome', 44: 'Sleep',
    45: 'sorry', 46: 'yes', 47: 'no', 48: 'thanks', 49: 'what',
    50: 'sad', 51: 'book'
}

prediction_started = False
last_prediction_time = 0
output = ""

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global prediction_started, last_prediction_time, output
    cap = cv2.VideoCapture(0)
    while True:
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks and prediction_started:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            current_time = time.time()

            if current_time - last_prediction_time > 7:
                if len(data_aux) == 84:  # Check if data_aux has 84 features
                    # Flatten and normalize data_aux to 42 features
                    flattened_data = data_aux[:42]  # Use the first 42 elements
                    normalized_data = [(item - min(flattened_data)) / (max(flattened_data) - min(flattened_data)) for item in flattened_data]
                    prediction = model.predict([np.asarray(normalized_data)])  # Predict using the normalized data
                else:
                    prediction = model.predict([np.asarray(data_aux)])  # Predict using the original data_aux

                predicted_character = labels_dict[int(prediction[0])]
                output += predicted_character
                last_prediction_time = current_time

                # Send AJAX request to update web page output (new)
                data = {'output': output}
                response = requests.post('http://localhost:5000/update_output', json=data)  # Use requests library

                # print("Current Prediction:", predicted_character)
                # print("Current Output:", output)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            # cv2.putText(frame, output, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_prediction():
    global prediction_started
    prediction_started = True
    return output

@app.route('/stop', methods=['POST'])
def stop_prediction():
    global prediction_started, output
    prediction_started = False
    translation = translator.translate(output, src='en', dest='te')
    translated_text = translation.text

    # Get current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

    # Save audio file with timestamp appended to its name
    audio_filename = f'output_{timestamp}.mp3'
    tts = gTTS(text=translated_text, lang='te')
    tts.save(audio_filename)

    # Play the latest audio file
    pygame.mixer.music.load(audio_filename)
    pygame.mixer.music.play()

    output = ""  # Reset output after translation
    return translated_text


@app.route('/backspace', methods=['POST'])
def backspace():
    global output
    output = output[:-1]
    return output

@app.route('/addspace', methods=['POST'])
def addspace():
    global output
    output = output+" "
    return output

@app.route('/update_output', methods=['GET'])
def update_output(): #suggestions
    global output, suggestion_words
    if output!="":
        search_word = output.split()[-1] if ' ' in output else output
        # print(search_word)
        suggestions = find_similar_words(search_word, suggestion_words)
        # print(suggestions)
    return jsonify({'output': output, 'suggestions': suggestions})


@app.route('/addsuggestion', methods=['POST'])
def addsuggestion():
    global output
    suggestion = request.json['suggestion']
    
    # Replace the last word in output with the clicked suggestion
    output_list = output.split()
    if output_list:
        output_list[-1] = suggestion
        output = " ".join(output_list)
    else:
        output = suggestion
    
    return output



if __name__ == "__main__":
    app.run(debug=True)
