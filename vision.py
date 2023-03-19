import mediapipe as mp
import cv2
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh


# adapted from: https://medium.com/mlearning-ai/iris-segmentation-mediapipe-python-a4deb711aae3

class Tracker:
    LEFT_IRIS = [474,475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    RIGHT_PUPIL = 473
    LEFT_PUPIL = 468

    N_LANDMARKS = 478

    # Left eye indices list
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    # Right eye indices list
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]


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

    def snap(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        self.frame_count += 1
        return frame

    def get_face_mesh(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame)
        
        if results.multi_face_landmarks[0].landmark is None:
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


    def draw(self, frame):
        left_iris, right_iris = self.mesh_points[self.LEFT_IRIS], self.mesh_points[self.RIGHT_IRIS]
        left_eye, right_eye = self.mesh_points[self.LEFT_EYE], self.mesh_points[self.RIGHT_EYE]
    
        # draw eyes and irises
        for color, points in zip(((0,0,255), (0,255,0)), ((left_iris, right_iris), (left_eye, right_eye))):
            for point in points:
                cv2.polylines(frame, [point], True, color, 1, cv2.LINE_AA)

        # draw every landmark
        for i in range(self.N_LANDMARKS):
            cv2.circle(frame, tuple(self.mesh_points[i]), 1, (255, 0, 0), -1)

        # draw pupils
        for points in (self.LEFT_PUPIL, self.RIGHT_PUPIL):
            cv2.circle(frame, tuple(self.mesh_points[points]), 4, (0, 0, 255), -1)

        # Calculate the FPS and display it on the output image
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    def __call__(self):
        frame = self.snap()
        self.get_face_mesh(frame)
        self.draw(frame)

        return frame

