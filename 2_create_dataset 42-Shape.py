import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './Data/Without Landmarks'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
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

            # Check the shape of the data_aux
            if len(data_aux) == 84:
                # Flatten and normalize data_aux to (42,) shape
                normalized_data = [(item - min(data_aux)) / (max(data_aux) - min(data_aux)) for item in data_aux]

                # Pad or truncate the data to ensure it has length 42
                if len(normalized_data) < 42:
                    normalized_data.extend([0] * (42 - len(normalized_data)))
                elif len(normalized_data) > 42:
                    normalized_data = normalized_data[:42]

                data.append(normalized_data)
                labels.append(dir_)
            elif len(data_aux) == 42:  # If already 42, append as it is
                data.append(data_aux)
                labels.append(dir_)

f = open('data_ASL.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
