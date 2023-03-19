import pyautogui
import numpy as np

class Controller:
    def __init__(self, gain=250) -> None:
        screen_w, screen_h = pyautogui.size()
        self.screen_center = (screen_w / 2, screen_h / 2)
        self.gain = gain

    def center_cursor(self):
        pyautogui.moveTo(self.screen_center[0], self.screen_center[1])


    def get_iris_offset(self, tracker, side):
        if side == "left":
            eye, iris = tracker.left_eye, tracker.left_iris
        elif side == "right":
            eye, iris = tracker.right_eye, tracker.right_iris

        # get the eye center
        (ex, ey) = np.mean(eye, axis=0)

        # get the iris center
        (ix, iy) = np.mean(iris, axis=0)

        delta_x = ix-ex
        delta_y = iy-ey
        return delta_x, delta_y

    def apply_controls(self, tracker):
        ldx, ldy = self.get_iris_offset(tracker, "left")
        rdx, rdy = self.get_iris_offset(tracker, "right")

        gaze_dx = (ldx + rdx) / 2
        gaze_dy = (ldy + rdy) / 2

        cursor_x = self.screen_center[0] + gaze_dx * self.gain
        cursor_y = self.screen_center[1] + gaze_dy * self.gain

        pyautogui.moveTo(cursor_x, cursor_y)