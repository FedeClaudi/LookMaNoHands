import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os.path as osp
import os
import cv2
import numpy as np
import pandas as pd

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the video mode:
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)


def get_landmarks_idxs(connections):
    return list(set(np.vstack(connections).ravel()))


LEFT_PUPIL = 468
RIGHT_PUPIL = 473

NOSE = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_NOSE)
MOUTH = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_LIPS)
OVAL = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_FACE_OVAL)
LEFT_EYE = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)
LEFT_IRIS = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_LEFT_IRIS)
RIGHT_IRIS = get_landmarks_idxs(mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS)


def extract_features(coords):
    center = np.mean(coords, axis=0)
    center[2] = np.max(coords[:, 2])

    # center coords
    coords -= center

    # get the position of the eyes, mouth and nose
    mouth = coords[MOUTH].mean(axis=0)
    nose = coords[NOSE].mean(axis=0)
    left_eye = coords[LEFT_EYE].mean(axis=0)
    right_eye = coords[RIGHT_EYE].mean(axis=0)
    left_iris = coords[LEFT_IRIS].mean(axis=0)
    right_iris = coords[RIGHT_IRIS].mean(axis=0)
    left_pupil = coords[LEFT_PUPIL]
    right_pupil = coords[RIGHT_PUPIL]

    # stack it into a vector
    features = np.array(
        [*mouth, *nose, *left_eye, *right_eye, *left_iris, *right_iris, *left_pupil, *right_pupil]
    )

    return features



# ---------------------------------------------------------------------------- #
#                                      RUN                                     #
# ---------------------------------------------------------------------------- #

# get all video files in data/raw
video_files = [osp.join("data/raw", f) for f in os.listdir("data/raw") if osp.isfile(osp.join("data/raw", f)) and f.endswith(".avi")]

    # loop over all video files
for video_file in video_files:
    with FaceLandmarker.create_from_options(options) as landmarker:
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

            if len(face_landmarker.face_landmarks) == 0:
                # set coords to nans
                coords = np.full((478, 3), np.nan)
            else:
                for (i, marker) in enumerate(face_landmarker.face_landmarks[0]):
                    coords[i, :] = [marker.x, marker.y, marker.z]

            # extract features
            features = extract_features(coords)

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
        df.to_csv(os.path.join(
            "./data", "processed", video_file.split("\\")[-1].replace("video.avi", ".csv")
        ), index=False)


    