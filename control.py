import pyautogui
import numpy as np
import tensorflow as tf

class Controller:


    LEFT = [474,475, 476, 477, 473, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
    RIGHT = [469, 470, 471, 472, 468, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
    FACE = [227, 116, 137, 123, 345, 346, 280, 352, 208, 199, 428]

    


    def __init__(self) -> None:
        self.model = tf.keras.models.load_model("my_mlp_model.h5")
        self.indices = [*self.LEFT, *self.RIGHT, *self.FACE]


    def __call__(self, tracker):
        x = tracker.mesh_points_normalized[self.indices].ravel()
        y = self.model.predict(x.reshape(1, -1), verbose=False)
        pyautogui.moveTo(y[0, 0], y[0, 1], duration=0, _pause=False)