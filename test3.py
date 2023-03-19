import mediapipe as mp
import cv2
import numpy as np
import time
from collections import namedtuple
import pyautogui

mp_face_mesh = mp.solutions.face_mesh


# from: https://medium.com/mlearning-ai/iris-segmentation-mediapipe-python-a4deb711aae3
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

RIGHT_PUPIL = 473
LEFT_PUPIL = 468

# Left eye indices list
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]# Right eye indices list
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

_eye_corners = namedtuple('eye_corners', 'inner outer ')
LEFT_CORNERS = _eye_corners(362, 263)
RIGHT_CORNERS = _eye_corners(133, 33)

N_LANDMARKS = 478

FACE_LEFT = [21, 102, 127]
FACE_RIGHT = [251, 389, 356]

screen_w, screen_h = pyautogui.size()
screen_center = (screen_w / 2, screen_h / 2)

GAIN = 250

def get_iris_position(mesh, side, z=40, d=2):
    if side == "left":
        iris_points = mesh[[*LEFT_IRIS, LEFT_PUPIL]]
        eye_points = mesh[LEFT_EYE]
        (Ax, Ay), (Bx, By) = mesh[LEFT_CORNERS.outer], mesh[LEFT_CORNERS.inner]
    else:   
        iris_points = mesh[[*RIGHT_IRIS, RIGHT_PUPIL]]
        eye_points = mesh[RIGHT_EYE]
        (Ax, Ay), (Bx, By) = mesh[RIGHT_CORNERS.inner], mesh[RIGHT_CORNERS.outer]


    # get the face extremities
    (flx, fly) = np.mean(mesh[FACE_LEFT], axis=0)
    (frx, fry) = np.mean(mesh[FACE_RIGHT], axis=0)

    # get the eye center
    (ex, ey) = np.mean(eye_points, axis=0)

    # get the iris center
    (ix, iy) = np.mean(iris_points, axis=0)

    delta_x = ix-ex
    delta_y = iy-ey
    return delta_x, delta_y
    
    






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

    x_calibration_readings, y_calibration_readings = [], []
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


        # track gaze
        mesh_points_normalized = np.array([[p.x, p.y] for p in results.multi_face_landmarks[0].landmark])
        l_dx, l_dy = get_iris_position(mesh_points_normalized, "left")
        r_dx, r_dy = get_iris_position(mesh_points_normalized, "right")
        cursor_dx = (l_dx + r_dx) / 2
        cursor_dy = (l_dy + r_dy) / 2

        if frame_count < 5:
            x_calibration_readings.append(cursor_dx)
            y_calibration_readings.append(cursor_dy)

            pyautogui.moveTo(screen_center[0], screen_center[1])

        elif frame_count == 5:
            x_calibration = np.mean(x_calibration_readings)
            y_calibration = np.mean(y_calibration_readings)

        else:
            cursor_dx -= x_calibration
            cursor_dy -= y_calibration

            print(f"{cursor_dx:.3f}  -- {cursor_dy:.3f}")
            pyautogui.moveTo(
                screen_center[0] + GAIN * screen_center[0]*cursor_dx, 
                screen_center[1] + GAIN * screen_center[1]*cursor_dy
            )

        # get landmarks coordinates in pixels
        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        left_iris, right_iris = mesh_points[LEFT_IRIS], mesh_points[RIGHT_IRIS]
        left_eye, right_eye = mesh_points[LEFT_EYE], mesh_points[RIGHT_EYE]



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

        # draw pupils
        for points in (LEFT_PUPIL, RIGHT_PUPIL):
            cv2.circle(frame, tuple(mesh_points[points]), 4, (255, 0, 0), -1)


        # Calculate the FPS and display it on the output image
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('a'):
            GAIN += 25
            



cap.release()
