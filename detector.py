import cv2
import dlib
import numpy as np
import time
# import matplotlib.pyplot as plt
# from filterpy.kalman import KalmanFilter
import pyautogui
from collections import deque

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords



class Tracker:
    left_idx = [36, 37, 38, 39, 40, 41] # keypoint indices for left eye
    right_idx = [42, 43, 44, 45, 46, 47] # keypoint indices for right eye

    def __init__(self, eye_th = 40, debug=True, avg_eye_pos=2, sensitivity=3):
        self.eye_th = eye_th
        self.face_detector = dlib.get_frontal_face_detector()
        try:
            self.shape_predictor = dlib.shape_predictor('shape_68.dat')
        except:
            print("shape_68.dat not found. Download it from: https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2")
            exit()

        self.debug = debug
        self.sensitivity = sensitivity

        self._lx, self._ly = deque(maxlen=avg_eye_pos), deque(maxlen=avg_eye_pos)
        self._rx, self._ry = deque(maxlen=avg_eye_pos), deque(maxlen=avg_eye_pos)


    def track(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale 
        small = cv2.resize(gray, (0, 0), fx=.5, fy=.5)

        rect = self.face_detector(small, 1)
        if len(rect) == 0:
            return
        rect = rect[0]

        shape = self.shape_predictor(small, rect)
        shape = shape_to_np(shape) * 2

        # detect pupil position        
        self.shape = shape
        self.gray = gray
        eyes = self.track_pupil()
        self.store_coords(*eyes)

        # draw shape
        if self.debug:
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            for eye in eyes:
                if eye is None:
                    continue
                cv2.circle(frame, eye[:2], int(eye[2]), (0, 255, 0), 2)

            cv2.circle(frame, (int(self.lx), int(self.ly)), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(self.rx), int(self.ry)), 5, (255, 0, 0), -1)

    @property
    def lx(self):
        lx = np.nanmean(self._lx)
        return lx if not np.isnan(lx) else 0
    
    @property
    def ly(self):
        ly = np.nanmean(self._ly)
        return ly if not np.isnan(ly) else 0
    
    @property
    def rx(self):
        rx = np.nanmean(self._rx)
        return rx if not np.isnan(rx) else 0
    
    @property
    def ry(self):
        ry = np.nanmean(self._ry)
        return ry if not np.isnan(ry) else 0

    def cut_eye(self, side):
        if side == "right":
            eye = self.shape[self.right_idx, :]
        else:
            eye = self.shape[self.left_idx, :]
        x_min, y_min = np.min(eye, axis=0)
        x_max, y_max = np.max(eye, axis=0)
        return self.gray[y_min:y_max, x_min:x_max], x_min, y_min


    def track_pupil(self):
        eyes = []
        for side in ("left", "right"):
            eye, _x, _y = self.cut_eye(side)

            _, eye = cv2.threshold(eye, self.eye_th, 255, cv2.THRESH_BINARY)

            # eye = cv2.dilate(eye, None, iterations=2)
            eye = cv2.erode(eye, None, iterations=4) 
            eye = cv2.medianBlur(eye, 5) 
            # cv2.imshow(side, eye)

            cnts, _ = cv2.findContours(255 - eye, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_NONE)
            if not len(cnts):
                eyes.append(None)
                continue
            cnt = max(cnts, key = cv2.contourArea) # finding contour with #maximum area
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            eyes.append((int(cx + _x), int(cy + _y), radius))
        return eyes


    def store_coords(self, left, right):
        self._lx.append(left[0] if left is not None else np.nan)
        self._ly.append(left[1] if left is not None else np.nan)
        self._rx.append(right[0] if right is not None else np.nan)
        self._ry.append(right[1] if right is not None else np.nan)

    def center_cursor(self):
        screenWidth, screenHeight = pyautogui.size()
        currentMouseX, currentMouseY = pyautogui.position()
        print(currentMouseX, currentMouseY)
        pyautogui.moveTo(int(screenWidth / 2), int(screenHeight / 2))


    def run(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0
        
        print("Stare at cursor to initialize")
        self.center_cursor()

        x, y = 0, 0
        while True:
            ret, frame = cap.read()
            self.track(frame)

            if frame_count > 5:
                dx = (x - self.lx) * self.sensitivity
                dy = (y - self.ly) * self.sensitivity
                # print("Moving cursor", dx, dy)
                # pyautogui.moveTo(int(self.lx), int(self.ly))
                pyautogui.moveRel(dx, -dy)

            # reset position if 'r' is pressed
            if cv2.waitKey(1) & 0xFF == ord('r'):
                print("Resetting cursor")
                self.center_cursor()

            # Calculate the FPS and display it on the output image
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            x, y = self.lx, self.ly
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    tracker = Tracker(sensitivity=70, avg_eye_pos=5)
    tracker.run()