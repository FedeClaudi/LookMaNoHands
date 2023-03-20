import mediapipe as mp
import cv2
import numpy as np
import time
from collections import namedtuple, deque
import tensorflow as tf
import pyautogui

mp_face_mesh = mp.solutions.face_mesh


# adapted from: https://medium.com/mlearning-ai/iris-segmentation-mediapipe-python-a4deb711aae3


def get_center(X, indices):
    if indices is not None:
        return np.mean(X[indices], axis=0)
    else:
        return np.mean(X, axis=0)

def get_length(X, i1, i2):
    A = X[i1]
    B = X[i2]

    d = np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    return d

class Tracker:
    LEFT_IRIS = [474,475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    LEFT_PUPIL = 468
    RIGHT_PUPIL = 473

    RIGHT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 133
    RIGHT_EYE_TOP = 159
    RIGHT_EYE_BOTTOM = 145

    LEFT_EYE_OUTER = 253
    LEFT_EYE_INNER = 362
    LEFT_EYE_TOP = 386
    LEFT_EYE_BOTTOM = 374

    N_LANDMARKS = 478

    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

    LEFT = [474,475, 476, 477, 473, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
    RIGHT = [469, 470, 471, 472, 468, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
    FACE = [227, 116, 137, 123, 345, 346, 280, 352, 208, 199, 428]

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # setup webcam streaming
        self.cap = cv2.VideoCapture(0)
        self.frame_count = -1
        self.start_time = time.time()

        ret, frame = self.cap.read()
        self.img_h, self.img_w, _ = frame.shape

        self.model = tf.keras.models.load_model("my_mlp_model.h5")
        self.indices = [*self.LEFT, *self.RIGHT, *self.FACE]

        # self.x, self.y = deque(maxlen=3), deque(maxlen=3)
        self.prev_x, self.prev_y = 0, 0

    def snap(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        self.frame_count += 1
        return frame

    def get_face_mesh(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # scale frame to speedup
        results = self.face_mesh.process(frame)
        
        if results.multi_face_landmarks is None:
            return 
        
        self.mesh_points_normalized = np.array([[p.x, p.y] for p in results.multi_face_landmarks[0].landmark])
        self.mesh_points = np.array([np.multiply([p.x, p.y], [self.img_w, self.img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

    @property
    def left_eye(self):
        return self.mesh_points_normalized[self.LEFT_EYE]
    
    @property
    def right_eye(self):
        return self.mesh_points_normalized[self.RIGHT_EYE]
    
    @property
    def left_iris(self):
        return self.mesh_points_normalized[[*self.LEFT_IRIS, self.LEFT_PUPIL]]
    
    @property
    def right_iris(self):
        return self.mesh_points_normalized[[*self.RIGHT_IRIS, self.RIGHT_PUPIL]]
    
    @property
    def fps(self):
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time


    def draw(self, frame):
        left_iris, right_iris = self.mesh_points[self.LEFT_IRIS], self.mesh_points[self.RIGHT_IRIS]
        left_eye, right_eye = self.mesh_points[self.LEFT_EYE], self.mesh_points[self.RIGHT_EYE]
    

        # draw every landmark
        for i in range(self.N_LANDMARKS):
            cv2.circle(frame, tuple(self.mesh_points[i]), 1, (255, 0, 0), -1)

        # draw eyes and irises
        for color, points in zip(((0,0,255), (0,255,0)), ((left_iris, right_iris), (left_eye, right_eye))):
            for point in points:
                cv2.polylines(frame, [point], True, color, 1, cv2.LINE_AA)

        # draw pupils
        for points in (self.LEFT_PUPIL, self.RIGHT_PUPIL):
            cv2.circle(frame, tuple(self.mesh_points[points]), 4, (0, 0, 255), -1)

        # Calculate the FPS and display it on the output image
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    """
    Extract informative features from tracking data.    
    """
    def extract_features(self):
        coords = self.mesh_points_normalized


        left_iris_center = get_center(coords, [*self.LEFT_IRIS, self.LEFT_PUPIL])
        right_iris_center = get_center(coords, [*self.RIGHT_IRIS, self.RIGHT_PUPIL])
        left_eye_center = get_center(coords, self.LEFT_EYE)
        right_eye_center = get_center(coords, self.RIGHT_EYE)

        face_center = get_center(coords, None)
        face_left = np.min(coords[:, 0])
        face_right = np.max(coords[:, 0])
        face_bottom = np.min(coords[:, 1])
        face_top = np.max(coords[:, 1])

        left_eye_width = get_length(coords, self.LEFT_EYE_INNER, self.LEFT_EYE_OUTER)
        right_eye_width = get_length(coords, self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER)
        left_eye_height = get_length(coords, self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM)
        right_eye_height = get_length(coords, self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM)

        eyes_width_ratio = left_eye_width / right_eye_width
        left_wh_ratio = left_eye_height/left_eye_width
        right_wh_ratio = right_eye_height/right_eye_width
        face_wh_ratio = (face_right-face_left)/(face_bottom-face_top)

        selected = [*self.LEFT_IRIS, self.LEFT_PUPIL, *self.RIGHT_IRIS, self.RIGHT_PUPIL, *self.LEFT_EYE, *self.RIGHT_EYE]

        features = np.hstack((
            # left_iris_center.ravel(), right_iris_center.ravel(),
            # left_eye_center.ravel(), right_eye_center.ravel(),
            # face_center.ravel(), 
            # face_left, face_right, face_bottom, face_top,
            # left_eye_width, right_eye_width,
            # left_eye_height, right_eye_height,
            # eyes_width_ratio,  
            # left_wh_ratio, right_wh_ratio, face_wh_ratio,
            coords[selected].ravel()
        ))

        return features




    def __call__(self, control=True):
        frame = self.snap()
        self.get_face_mesh(frame)
        self.draw(frame)

        if not control:
            return

        # move cursor
        y = self.model.predict(self.extract_features().reshape(1, -1), verbose=False)

        new_x, new_y = y[0, 0], y[0, 1]
        # curr_x, curr_y = pyautogui.position()
        print(new_x, new_y)

        # distance
        # dist = np.sqrt((new_x - self.prev_x)**2 + (new_y - self.prev_y)**2)

        # if dist > 100:
        pyautogui.moveTo(new_x, new_y, duration=0.1, _pause=False)
            # self.prev_x, self.prev_y = new_x, new_y


        # self.x.append(y[0, 0])
        # self.y.append(y[0, 1])

        # if self.frame_count > 3:
        #     pyautogui.moveTo(np.mean(self.x), np.mean(self.y), duration=0, _pause=False)

        return frame

