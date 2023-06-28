import mediapipe as mp
import cv2
import numpy as np
import time
from collections import namedtuple, deque
# import tensorflow as tf
import pyautogui
from face_geometry import PCF, get_metric_landmarks, procrustes_landmark_basis

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh_connections = mp.solutions.face_mesh_connections

drawing_spec = dict(
    face = (mp_drawing.DrawingSpec(thickness=3, circle_radius=1, color=(255, 255, 255)),
        (mp_face_mesh_connections.FACEMESH_FACE_OVAL, )),
    eye=(mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(0, 0, 255)), 
        (mp_face_mesh_connections.FACEMESH_LEFT_EYE,
        mp_face_mesh_connections.FACEMESH_RIGHT_EYE,
         )),
    iris=(mp_drawing.DrawingSpec(thickness=2, circle_radius=0, color=(0, 255, 0)), 
        (mp_face_mesh_connections.FACEMESH_LEFT_IRIS,
        mp_face_mesh_connections.FACEMESH_RIGHT_IRIS,
         )),
)

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

points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

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
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9,
            static_image_mode=False,
        )

        # setup webcam streaming
        self.cap = cv2.VideoCapture(0)

        # set video fps to 60
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # set video resolution to 1280x720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.frame_count = -1
        self.start_time = time.time()

        ret, frame = self.cap.read()
        self.img_h, self.img_w, _ = frame.shape
        
        # pseudo camera internals
        focal_length = self.img_w
        center = (self.img_w / 2, self.img_h / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )
        self.dist_coeff = np.zeros((4, 1))
        self.pcf = PCF(
        near=1,
        far=10000,
        frame_height=self.img_h,
        frame_width=self.img_w,
        fy=self.camera_matrix[1, 1],
    )

        # gaze tracking model
        # self.model = tf.keras.models.load_model("my_mlp_model.h5")
        self.indices = [*self.LEFT, *self.RIGHT, *self.FACE]

        # cursor control
        self.prev_x, self.prev_y = 0, 0


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
        # Calculate the FPS and display it on the output image
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for (spec, idxs) in drawing_spec.values():
            for idx in idxs:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=self._mesh,
                    connections=idx,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=spec,
                )

        # # draw every landmark
        # for i in range(self.N_LANDMARKS):
        #     cv2.circle(frame, tuple(self.mesh_points[i]), 1, (255, 255, 255), -1)

        # draw offset nose position for head transform info
        nose_tip = self.model_points[0]
        nose_tip_extended = 2.5 * self.model_points[0]
        (nose_pointer2D, jacobian) = cv2.projectPoints(
            np.array([nose_tip, nose_tip_extended]),
            self.mp_rotation_vector,
            self.mp_translation_vector,
            self.camera_matrix,
            self.dist_coeff,
        )

        nose_tip_2D, nose_tip_2D_extended = nose_pointer2D.squeeze().astype(int)
        cv2.line(
            frame, nose_tip_2D, nose_tip_2D_extended, (0, 0, 255), 4
        )




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
        # face_left = np.min(coords[:, 0])
        # face_right = np.max(coords[:, 0])
        # face_bottom = np.min(coords[:, 1])
        # face_top = np.max(coords[:, 1])

        # left_eye_width = get_length(coords, self.LEFT_EYE_INNER, self.LEFT_EYE_OUTER)
        # right_eye_width = get_length(coords, self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER)
        # left_eye_height = get_length(coords, self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM)
        # right_eye_height = get_length(coords, self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM)

        # eyes_width_ratio = left_eye_width / right_eye_width
        # left_wh_ratio = left_eye_height/left_eye_width
        # right_wh_ratio = right_eye_height/right_eye_width
        # face_wh_ratio = (face_right-face_left)/(face_bottom-face_top)


        features = np.hstack((
            left_iris_center.ravel(), right_iris_center.ravel(),
            left_eye_center.ravel(), right_eye_center.ravel(),
            face_center.ravel(), 
            # self.head_transform.ravel(),
            # face_left, face_right, face_bottom, face_top,
            # left_eye_width, right_eye_width,
            # left_eye_height, right_eye_height,
            # eyes_width_ratio,  
            # left_wh_ratio, right_wh_ratio, face_wh_ratio,
            # coords[selected].ravel()
        ))

        # features = coords[selected].ravel()

        return features


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
            return False

        self._mesh = results.multi_face_landmarks[0]
        self.mesh_points_normalized = np.array([[p.x, p.y] for p in results.multi_face_landmarks[0].landmark])
        self.mesh_points = np.array([np.multiply([p.x, p.y], [self.img_w, self.img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        return True


    def get_head_transform(self):
        landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in self._mesh.landmark]
                ).T[:, :468]
        
        metric_landmarks, pose_transform_mat = get_metric_landmarks(
                    landmarks.copy(), self.pcf
                )
        self.model_points = metric_landmarks[0:3, points_idx].T

        # see here:
        # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
        pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
        self.mp_rotation_vector, _ = cv2.Rodrigues(pose_transform_mat[:3, :3])
        self.mp_translation_vector = pose_transform_mat[:3, 3, None]
        self.head_transform = pose_transform_mat



    def __call__(self, control=True):
        frame = self.snap()
        self.get_face_mesh(frame)
        self.get_head_transform()
        self.draw(frame)

        if not control:
            return frame

        # move cursor
        y = self.model.predict(self.extract_features().reshape(1, -1), verbose=False)
        new_x, new_y = y[0, 0], y[0, 1]
        pyautogui.moveTo(new_x, new_y, duration=0.1, _pause=False)


        return frame

