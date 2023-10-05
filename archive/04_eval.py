import mediapipe as mp
import time
import cv2
import numpy as np
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import archive.utils as utils
import model

# https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python#live-stream

# define a global variable to store the results
results = None
model_type="MLP"

# Create a face landmarker instance with the live stream mode:
def store_results(new_result: utils.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global results
    results = new_result
    

options = utils.FaceLandmarkerOptions(
    base_options=utils.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=utils.VisionRunningMode.LIVE_STREAM,
    result_callback=store_results,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    )

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_type == "MLP":
    network = model.MLP(utils.N_FEATURES, 2)
    network.load_state_dict(torch.load("models/mlp.pt"))
else:
    network = model.LSTM(utils.N_FEATURES, 128, 2)
    network.load_state_dict(torch.load("models/rnn.pt"))

network.eval()
network.to(device)

with utils.FaceLandmarker.create_from_options(options) as landmarker:
    # Set up video capture from default camera
    cap = cv2.VideoCapture(0)

    # set video fps to 60
    cap.set(cv2.CAP_PROP_FPS, 60)
    print("Starting")

    # Set up FPS counter
    frames = 0
    start_time = time.time()
    coords = np.zeros((478, 3))

    #   The landmarker is initialized. Use it here.
    hidden = None
    while True:
        ret, frame = cap.read()
        # Display frame
        frames += 1

        # Calculate FPS
        if frames > 10:
            fps = round(frames / (time.time() - start_time), 2)
        else:
            fps = 0
        frame = cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Send live image data to perform face landmarking.
        frame_timestamp_ms = int(round(time.time() * 1000))
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        
        if results is not None:
            if len(results.face_landmarks) == 0:
                print("No face detected")
                continue

            frame = utils.draw_landmarks_on_image(frame, results)

            # check if a recognized action is being made
            for blend in results.face_blendshapes[0]:
                if blend.score > 0.5 and  "eye" not in blend.category_name:
                    print(blend.category_name, blend.score)

            # extract features
            coords = utils.extract_landmark_coords(results, coords)
            features = utils.extract_features(coords)
            # convert to tensor
            features = torch.from_numpy(features).float().to(device)

            if model_type == "MLP":
                output = network(features.reshape(1, -1))
            else:
                output, hidden = network(features.reshape(1, -1), hidden=hidden)

            x, y = output.ravel().detach().cpu().numpy().astype(int)
            print(f"X: {x}, Y: {y}")
            pyautogui.moveTo(max(1, x), max(1, y), duration=0.0, _pause=False)


        # show frame
        # rescale to 3x
        frame = cv2.resize(frame, (0, 0), fx=3, fy=3)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        