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

    def __init__(self, eye_th = 40, debug=True, avg_eye_pos=2):
        self.eye_th = eye_th
        self.face_detector = dlib.get_frontal_face_detector()
        try:
            self.shape_predictor = dlib.shape_predictor('shape_68.dat')
        except:
            print("shape_68.dat not found. Download it from: https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2")
            exit()

        self.debug = debug
    
        self._lx, self._ly = deque(maxlen=avg_eye_pos), deque(maxlen=avg_eye_pos)
        self._rx, self._ry = deque(maxlen=avg_eye_pos), deque(maxlen=avg_eye_pos)

    """
    Get the coordinates of the keypoints for the left/right eye
    """
    def _get_eye(self, side):
        if side == "right":
            eye = self.shape[self.right_idx, :]
        else:
            eye = self.shape[self.left_idx, :]
        return eye

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


    """
    Get position of the eyes and pupils
    """
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

            # show each eye at the top left corner of the frame
            for i, side in enumerate(("left", "right")):
                eye_img = self.cut_eye(side)[0]
                eye_img = self.eye_img_processing(eye_img)
                eye_img = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2BGR)
                h, w, _ = eye_img.shape
                frame[:h, i*w:(i+1)*w] = eye_img
                




    """
    Cut eye from frame
    """
    def cut_eye(self, side):
        eye = self._get_eye(side)
        x_min, y_min = np.min(eye, axis=0)
        x_max, y_max = np.max(eye, axis=0)
        return self.gray[y_min:y_max, x_min:x_max], x_min, x_max, y_min, y_max
    
    def eye_img_processing(self, eye_im):
        _, eye_im = cv2.threshold(eye_im, self.eye_th, 255, cv2.THRESH_BINARY)
        # eye_im = cv2.dilate(eye_im, None, iterations=2)
        eye_im = cv2.erode(eye_im, None, iterations=4) 
        eye_im = cv2.medianBlur(eye_im, 5) 
        # cv2.imshow(side, eye_im)
        return eye_im
        
    """
    Get pupil position
    """
    def track_pupil(self):
        eyes = []
        for side in ("left", "right"):
            eye_im, x_min, x_max, y_min, y_max = self.cut_eye(side)
            eye_im = self.eye_img_processing(eye_im)

            cnts, _ = cv2.findContours(255 - eye_im, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_NONE)
            if not len(cnts):
                eyes.append(None)
                continue
            cnt = max(cnts, key = cv2.contourArea) # finding contour with #maximum area
            (px, pq), radius = cv2.minEnclosingCircle(cnt)

            # get the coordinates relative to the center of the eye_im
            eye_width = (x_max - x_min)/2
            eye_height = eye_width / 3

            centered_x = (px - eye_width)/eye_width
            centered_y = (pq - eye_height)/eye_height

            eyes.append((int(px + x_min), int(pq + y_min), radius, centered_x, centered_y))
        return eyes


    def store_coords(self, left, right):
        self._lx.append(left[3] if left is not None else np.nan)
        self._ly.append(left[4] if left is not None else np.nan)
        self._rx.append(right[3] if right is not None else np.nan)
        self._ry.append(right[4] if right is not None else np.nan)

    """
        Get the center of each eye by getting the outer/inner most point of each eye and taking
        the average of the coordinates of each eye's points and then average
    """
    def get_eyes_center(self):
        left_eye = self._get_eye("left")
        right_eye = self._get_eye("right")
        lx, ly = np.mean(left_eye, axis=0)
        rx, ry = np.mean(right_eye, axis=0)
        x = (lx + rx) / 2
        y = (ly + ry) / 2
        return x, y



class Controller:
    def __init__(self, *args, sensitivity=100, **kwargs):
        self.sensitivity = sensitivity
        self.screen_distance = 40 # cm
        self.tracker = Tracker(*args, **kwargs)

        screen_w, screen_h = pyautogui.size()
        self.screen_center = (screen_w / 2, screen_h / 2)

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        frame_h, frame_w, _ = frame.shape

        # get the w/h scaling ratios to go from frame coordinates to screen coordinates
        self.w_scaling = screen_w / frame_w
        self.h_scaling = screen_h / frame_h
        

    def center_cursor(self):
        pyautogui.moveTo(*self.screen_center)

    def move_cursor(self):
        x, y = self.tracker.get_eyes_center()

        left = self.tracker.lx, self.tracker.ly
        right = self.tracker.rx, self.tracker.ry
        print(f"Left - x: {left[0]:.2f}, y: {left[1]:.2f} | Right - x: {right[0]:.2f}, y: {right[1]:.2f}")

        delta_x_left = self.screen_distance * left[0]
        delta_y_left = self.screen_distance * left[1]
        delta_x_right = self.screen_distance * right[0]
        delta_y_right = self.screen_distance * right[1]

        delta_x= (delta_x_left + delta_x_right) / 2 * self.sensitivity
        delta_y= (delta_y_left + delta_y_right) / 2 * self.sensitivity

        pyautogui.moveTo((x + delta_x) * self.w_scaling, (y - delta_y) * self.h_scaling)



    def run(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)


        while True:
            ret, frame = cap.read()

            # flip the frame so it's not the mirror view
            frame = cv2.flip(frame, 1)

            self.tracker.track(frame)

            if frame_count > 5:
                self.move_cursor()

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


        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    ctrl = Controller(sensitivity=60, avg_eye_pos=2, eye_th=30)
    ctrl.run()