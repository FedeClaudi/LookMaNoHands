import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os.path as osp
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

import utils

# define a global variable to store the results
results = None

# Create a face landmarker instance with the live stream mode:
def store_data(new_result, output_image: mp.Image, timestamp_ms: int):
    global results
    results = new_result
    

options = utils.FaceLandmarkerOptions(
    base_options=utils.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=utils.VisionRunningMode.LIVE_STREAM,
    result_callback=store_data,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    )



def get_landmarks_idxs(connections):
    return list(set(np.vstack(connections).ravel()))



def draw_connection(ax, m1, m2, **kwargs):
    ax.plot(
            [m1[0], m2[0]],
            [m1[1], m2[1]],
            [m1[2], m2[2]],
            **kwargs
        )



class Head:
    def __init__(self, coords):
        # get the center
        self.center = coords.mean(axis=0)
        self.center[2] = np.max(coords[:, 2])

        # get the center of each face element
        self.nose = coords[utils.NOSE].mean(axis=0)
        self.mouth = coords[utils.MOUTH].mean(axis=0)
        self.oval = coords[utils.OVAL].mean(axis=0)
        self.left_eye = coords[utils.LEFT_EYE].mean(axis=0)
        self.right_eye = coords[utils.RIGHT_EYE].mean(axis=0)

    def draw(self, ax):
        ax.scatter(self.center[0], self.center[1], self.center[2], c='k', alpha=1, s=20)
        draw_connection(ax, self.center, self.nose, c='k', alpha=1, linewidth=2)
        draw_connection(ax, self.center, self.mouth, c='k', alpha=1, linewidth=2)
        draw_connection(ax, self.center, self.left_eye, c='k', alpha=1, linewidth=2)
        draw_connection(ax, self.center, self.right_eye, c='k', alpha=1, linewidth=2)
        draw_connection(ax, self.left_eye, self.right_eye, c='k', alpha=1, linewidth=2)
        draw_connection(ax, self.left_eye, self.mouth, c='k', alpha=1, linewidth=2)
        draw_connection(ax, self.right_eye, self.mouth, c='k', alpha=1, linewidth=2)



# get all video files in data/raw
video_files = [osp.join("data/raw", f) for f in os.listdir("data/raw") if osp.isfile(osp.join("data/raw", f)) and f.endswith(".avi")]

# start an interactive 3d plot
fig = plt.figure( figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d', xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))


plt.ion()

with utils.FaceLandmarker.create_from_options(options) as landmarker:

    # Set up video capture from default camera
    cap = cv2.VideoCapture(0)

    # set video fps to 60
    cap.set(cv2.CAP_PROP_FPS, 60)

    # loop over frames
    framen = 0
    coords = np.zeros((478, 3))

    while True:
        # read
        ret, frame = cap.read()
        if not ret:
            break
        framen += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # process
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(round(time.time() * 1000))
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if results is not None:
            if len(results.face_landmarks) == 0:
                continue

            for (i, marker) in enumerate(results.face_landmarks[0]):
                coords[i, :] = [marker.x, marker.y, marker.z]


            ax.clear()
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='b', s=1)
            ax.scatter(coords[utils.eyes_landmark, 0], coords[utils.eyes_landmark, 1], coords[utils.eyes_landmark, 2], c='k', alpha=1, s=10)
            ax.scatter(coords[utils.FaceLandmarkerOptionsface_landmarks, 0], coords[utils.FaceLandmarkerOptionsface_landmarks, 1], coords[utils.FaceLandmarkerOptionsface_landmarks, 2], c='r', alpha=0.5, s=8)
            

            # scatter a point at each corner of a qube going from the origin to (1, 1, 1)
            ax.scatter([0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1], c='k', s=1)

            head = Head(coords)
            head.draw(ax)
            plt.draw()
            plt.pause(0.001)


plt.ioff()


    