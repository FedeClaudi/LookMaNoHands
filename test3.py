import mediapipe as mp
import cv2
import numpy as np
import time
from collections import namedtuple

mp_face_mesh = mp.solutions.face_mesh


# from: https://medium.com/mlearning-ai/iris-segmentation-mediapipe-python-a4deb711aae3
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Left eye indices list
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]# Right eye indices list
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

_eye_corners = namedtuple('eye_corners', 'inner outer ')
LEFT_CORNERS = _eye_corners(362, 263)
RIGHT_CORNERS = _eye_corners(133, 33)

N_LANDMARKS = 478

def get_iris_position(iris_points, eye_points):
    # get the iris position "within" the eye as values between -1 and 1

    eye_center = np.mean(eye_points, axis=0)
    iris_center = np.mean(iris_points, axis=0)
    delta = iris_center - eye_center

    # normalize based on eye width
    eye_width = np.linalg.norm(eye_points[0] - eye_points[3])
    return delta / eye_width





with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as face_mesh:
    


    # For webcam input:
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frame_count = 0


    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)#getting width and height or frame
        img_h, img_w = frame.shape[:2]

        if results is None or results.multi_face_landmarks is None:
            continue

        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

        left_iris, right_iris = mesh_points[LEFT_IRIS], mesh_points[RIGHT_IRIS]
        left_eye, right_eye = mesh_points[LEFT_EYE], mesh_points[RIGHT_EYE]

        # get the iris position "within" the eye
        left = get_iris_position(left_iris, left_eye)
        right = get_iris_position(right_iris, right_eye)


        # left_position, right_position = get_iris_position(left_iris, left_eye), get_iris_position(right_iris, right_eye)
        # print(f"Left position {left_position} - Right position {right_position}")

        # draw eyes and irises
        for color, points in zip(((0,0,255), (0,255,0)), ((left_iris, right_iris), (left_eye, right_eye))):
            for point in points:
                cv2.polylines(frame, [point], True, color, 1, cv2.LINE_AA)

        # draw every landmark
        for i in range(N_LANDMARKS):
            cv2.circle(frame, tuple(mesh_points[i]), 1, (255, 0, 0), -1)

        # draw eye corners
        for points in (LEFT_CORNERS, RIGHT_CORNERS):
            cv2.circle(frame, tuple(mesh_points[points.inner]), 2, (0, 0, 255), -1)
            cv2.circle(frame, tuple(mesh_points[points.outer]), 2, (0, 255, 0), -1)



        # Calculate the FPS and display it on the output image
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
