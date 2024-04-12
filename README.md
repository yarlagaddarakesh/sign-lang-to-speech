Sign Language to Speech (Telugu)
Overview
This project aims to convert Telugu sign language into speech using machine learning techniques. It consists of several steps:

Steps:
Step-1: Collect Images

    Run 1_collect_imgs.py file.
    Adjust the no.of classes variable based on your requirements.
    Set the no.of users variable to specify the number of members from whom you want to collect images.
    For each class, the script prompts you to enter the number of hands you want to capture (1 or 2).
    It creates a Data Directory containing two subfolders:
    with landmarks: Images with landmarks to assess landmark detection.
    without landmarks: Images without landmarks.

Step-2: Create Dataset

    Use 2_create_dataset_42_shape.py to create a dataset from the collected images.
    This script generates a dataset with 42 shapes and creates a pickle file to store the landmarks.

Step-3: Train Model

    Execute 3_train_model.py to create a Random Forest model using the pickle data file.
    After execution, a model.p file containing the trained model will be generated.

Step-4: Real-time Prediction

    Run 4_realtime_prediction.py for real-time prediction.
    The script opens the camera and starts detecting signs.
    It waits for 10 seconds after each detection.
    Press 't' to stop the prediction and convert the signs into speech in Telugu language.
    Note:
    Ensure your environment is set up correctly with required libraries.
    Fine-tune parameters as needed for optimal performance.

Example Usage:
    python 1_collect_imgs.py
    python 2_create_dataset_42_shape.py
    python 3_train_model.py
    python 4_realtime_prediction.py

Requirements:
mediapipe
opencv-python
scikit-learn
numpy
gtts
pygame
googletrans==4.0.0-rc1

