import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os.path as osp
import os
import cv2
import numpy as np
import pandas as pd

import utils


# Create a face landmarker instance with the video mode:
options = utils.FaceLandmarkerOptions(
    base_options=utils.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=utils.VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)




# ---------------------------------------------------------------------------- #
#                                      RUN                                     #
# ---------------------------------------------------------------------------- #

# get all video files in data/raw
video_files = [osp.join("data/raw", f) for f in os.listdir("data/raw") if osp.isfile(osp.join("data/raw", f)) and f.endswith(".avi")]

    # loop over all video files
for video_file in video_files:
    save_path = os.path.join(
            "./data", "processed", video_file.split("\\")[-1].replace("video.avi", ".csv")
        )
    
    # skip if already processed
    if osp.isfile(save_path):
        print(f"Skipping {video_file}")
        continue

    with utils.FaceLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    
        # open cap and get FPS
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Opened: {video_file} with FPS: {fps}")

        # loop over frames
        framen = 0
        coords = np.zeros((478, 3))

        data = dict(
            cursor_x = [],
            cursor_y = [],
        )
        for i in np.arange(24):
            data[f"feature_{i}"] = []


        while cap.isOpened():
            # read
            ret, frame = cap.read()
            if not ret:
                break
            framen += 1

            # process
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = int(framen * 1000 / fps)
            face_landmarker = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # extract features
            coords = utils.extract_landmark_coords(face_landmarker, coords)
            features = utils.extract_features(coords)

            # add features to data
            for (i, feature) in enumerate(features):
                data[f"feature_{i}"].append(feature)

        # load the corresponding cursor npz data
        cursor_data = np.load(video_file.replace("video.avi", "cursor_pos.npy"))

        # assert same length
        assert len(cursor_data) == len(data["feature_0"])

        # add cursor data to data
        data["cursor_x"] = cursor_data[:, 0]
        data["cursor_y"] = cursor_data[:, 1]

        # save data as csv
        df = pd.DataFrame(data)

        # interpolate missing values
        df = df.interpolate(method="linear", limit_direction="both")

        # smooth
        df = df.rolling(window=5, min_periods=1).mean()

        df.to_csv(save_path, index=False)


    