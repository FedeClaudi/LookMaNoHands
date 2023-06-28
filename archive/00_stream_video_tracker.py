
import cv2
import time
import numpy as np

import vision

tracker = vision.Tracker()



while True:
    frame = tracker(control=False)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
