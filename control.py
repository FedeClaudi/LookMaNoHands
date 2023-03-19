import pyautogui
import numpy as np
from collections import deque

class Controller:
    def __init__(self, gain=250, T=10) -> None:
        screen_w, screen_h = pyautogui.size()
        self.screen_center = (screen_w / 2, screen_h / 2)
        self.gain = gain

        # initialize Kalman filter
        self._x, self._y = deque(maxlen=T), deque(maxlen=T)
        self.T = T

    def center_cursor(self):
        pyautogui.moveTo(self.screen_center[0], self.screen_center[1], duration=0, _pause=False)


    def get_iris_offset(self, tracker, side):
        if side == "left":
            eye, iris = tracker.left_eye, tracker.left_iris
            (Ax,Ay), (Bx, By) = tracker.mesh_points_normalized[tracker.LEFT_CORNERS.inner], tracker.mesh_points_normalized[tracker.LEFT_CORNERS.outer]
        elif side == "right":
            eye, iris = tracker.right_eye, tracker.right_iris
            (Ax,Ay), (Bx, By) = tracker.mesh_points_normalized[tracker.RIGHT_CORNERS.inner], tracker.mesh_points_normalized[tracker.RIGHT_CORNERS.outer]

        # get the eye center
        (ex, _) = np.mean(eye, axis=0)

        # get the iris center
        (ix, iy) = np.mean(iris, axis=0)

        # get the distance between the iris and the line between A, B
        h = np.abs((Bx-Ax)*(Ay-iy) - (Ax-ix)*(By-Ay)) / np.sqrt((Bx-Ax)**2 + (By-Ay)**2)

        

        delta_x = ix-ex
        delta_y = h
        return delta_x, delta_y

    def apply_controls(self, tracker):
        ldx, ldy = self.get_iris_offset(tracker, "left")
        rdx, rdy = self.get_iris_offset(tracker, "right")

        gaze_dx = (ldx + rdx) / 2
        gaze_dy = (ldy + rdy) / 2

        cursor_x = self.screen_center[0] + gaze_dx * self.gain
        cursor_y = self.screen_center[1] + gaze_dy * self.gain

        # update deques
        self._x.append(cursor_x)
        self._y.append(cursor_y)

        x, y = np.mean(self._x), np.mean(self._y)

        if tracker.frame_count < self.T:
            self.center_cursor()
        else:
            if tracker.frame_count == self.T:
                
                self.x_calibration = np.mean(self._x)
                self.y_calibration = np.mean(self._y)
                print(f"Calibration complete - x: {self.x_calibration:.2f}, y: {self.y_calibration:.2f}")
                return


            x -= self.x_calibration
            y -= self.y_calibration

            pyautogui.moveTo(self.screen_center[0] + x, 
                             self.screen_center[1] - y, 
                             duration=0, _pause=False)