import os
import cv2
import random
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

DATA_DIR = './Data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

WITH_LANDMARKS_DIR = os.path.join(DATA_DIR, 'With Landmarks')
if not os.path.exists(WITH_LANDMARKS_DIR):
    os.makedirs(WITH_LANDMARKS_DIR)

WITHOUT_LANDMARKS_DIR = os.path.join(DATA_DIR, 'Without Landmarks')
if not os.path.exists(WITHOUT_LANDMARKS_DIR):
    os.makedirs(WITHOUT_LANDMARKS_DIR)

number_of_classes = 36
users_per_class = 3

cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    for j in range(number_of_classes):
        class_dir_with_landmarks = os.path.join(WITH_LANDMARKS_DIR, str(j))
        class_dir_no_landmarks = os.path.join(WITHOUT_LANDMARKS_DIR, str(j))

        if not os.path.exists(class_dir_with_landmarks):
            os.makedirs(class_dir_with_landmarks)
        if not os.path.exists(class_dir_no_landmarks):
            os.makedirs(class_dir_no_landmarks)

        
        print('Collecting data for class {}'.format(j))
        landmarks_count = int(input("Enter the landmarks count for class {} (1 or 2): ".format(j)))
        while landmarks_count not in [1, 2]:
            print("Invalid input. Please enter 1 or 2.")
            landmarks_count = int(input("Enter the landmarks count for class {} (1 or 2): ".format(j)))


        for user in range(users_per_class):
            user_dir = os.path.join(DATA_DIR, str(j))

            print(f'Collecting data for user {user + 1} in class {j + 0}')
            user_dir_with_landmarks = class_dir_with_landmarks
            user_dir_no_landmarks = class_dir_no_landmarks
            

            # Display present user number and capturing class number
            while True:
                ret, frame = cap.read()
                user_text = f'User: {user + 1}, Class: {j}'
                cv2.putText(frame, user_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) == ord('q'):
                    break

            # Capture a random number of images between 45 and 60
            image_count = random.randint(55, 75)
            counter = 0

            # Find the last serial number used in the class
            existing_serial_numbers = [int(filename.split('.')[0]) for filename in os.listdir(user_dir_no_landmarks)]
            last_serial_number = max(existing_serial_numbers) + 1 if existing_serial_numbers else 0

            while counter < image_count:
                ret, frame = cap.read()
                cv2.imshow('frame', frame)
                cv2.waitKey(25)

                # Convert the image to RGB before processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the image with MediaPipe Hands to detect landmarks
                results = hands.process(frame_rgb)

                # Check if landmarks are detected and count matches user input
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == landmarks_count:
                    cv2.imwrite(os.path.join(user_dir_no_landmarks, f'{last_serial_number}.jpg'), frame)
                    last_serial_number += 1
                    counter += 1

cap.release()
cv2.destroyAllWindows()
