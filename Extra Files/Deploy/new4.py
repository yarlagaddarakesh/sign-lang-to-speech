import time
import pickle
import streamlit as st
from gtts import gTTS
import pygame
import cv2
import mediapipe as mp
import numpy as np
from googletrans import Translator
import requests
from bs4 import BeautifulSoup

# Initialize Pygame for audio playback
pygame.mixer.init()
translator = Translator()

model_dict = pickle.load(open('./model.p', 'rb'))
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
    36: 'Home', 37: 'Brother', 38: 'Pay', 39: 'Justice', 40: 'Science',
    41: 'Please', 42: 'Hello', 43: 'You are Welcome', 44: 'Good Bye',
    45: 'Sorry', 46: 'Yes', 47: 'No', 48: 'Thanks', 49: 'What',
    50: 'Sad', 51: ' '
}

def get_related_words(predicted_character):
    url = f"https://www.google.com/search?q=define+{predicted_character}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    related_words = []
    for result in soup.select(".lr_dct_ent > .lr_dct_sf_h"):
        related_words.append(result.text.strip())
    return related_words

# Function to display suggested words
def display_suggested_words(predicted_character):
    st.subheader("Suggested Words:")
    related_words = get_related_words(predicted_character)
    if related_words:
        st.write(related_words[:3])  # Display the first 3 related words
    else:
        st.write("No suggested words found.")


# Function to translate text and play audio
def translate_and_play(text):
    filename = 'output.mp3'        
    if text:
        st.write("Predicted text:", text)
        translation = translator.translate(text, src='en', dest='te')
        translated_text = translation.text
        tts = gTTS(text=translated_text, lang='te')
        tts.save(filename)  # Save audio with a unique filename
        st.success("Translated Text: {}".format(translated_text))
        audio_file = open('output.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3', start_time=0)

    else:
        st.warning("No text detected for translation.")



# Streamlit app
def main():
    output = ""  # Initialize output
    st.title("Sign Language Translation")
    start_button = st.button("Sign Language Input")
    backspace_button = st.button("Backspace")

    if start_button:
        cap = cv2.VideoCapture(0)
        output_text = st.empty()
        stframe = st.empty()
        last_prediction_time = time.time()

        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()

            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
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

                if current_time - last_prediction_time > 2:
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

                    output_text.text(output)  # Update output_text with the latest value of output

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, output, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            stframe.image(frame, channels="BGR")
            key = cv2.waitKey(1)
            if time.time() - last_prediction_time > 5:
                output_text.empty()
                translate_and_play(output)
                stframe.empty()  # Clear the frame after translation and audio playback
                break

            if backspace_button:  # Check if backspace button is pressed
                output = output[:-1]  # Remove the last character from output

        display_suggested_words(output)  # Call the function outside the loop

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
