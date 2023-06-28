import numpy as np
import pyautogui
import time
import cv2
import cv2
from datetime import datetime


cap = cv2.VideoCapture(0)

snaps_per_frame = 2


# ---------------------------------------------------------------------------- #
#                                 GENERATE DATA                                #
# ---------------------------------------------------------------------------- #
screen_w, screen_h = pyautogui.size()

# select N random points on the screen
# x_pts = np.random.randint(5, screen_w-5, size=(T-4))
# y_pts = np.random.randint(5, screen_h-5, size=(T-4))
# pts = np.vstack((x_pts, y_pts)).T

# add 4 corners to points
all_pts = []
for delta in (10, 50, 75, 125, 150, 200, 250, 300):

    # make an array with 4 corners and midpoints between corners
    corners = np.array([
        [delta, delta],
        [delta, screen_h/2],
        [delta, screen_h - delta],
        [screen_w/2, screen_h - delta],
        [screen_w - delta, screen_h - delta],
        [screen_w - delta, screen_h/2],
        [screen_w - delta, delta],
        [screen_w/2, delta],
        [screen_w/2, screen_h/2]
    ])

    # make pts as 10 repetitions of corners
    pts = np.tile(corners, (2, 1))
    all_pts.append(pts)
pts = np.vstack(all_pts)

X = np.zeros((snaps_per_frame * pts.shape[0], features.shape[0]))
Y = np.zeros((snaps_per_frame * pts.shape[0], 2))

# center cursor
pyautogui.moveTo(screen_w / 2, screen_h / 2, duration=0, _pause=False)
time.sleep(0.5)


# move through each point
j = 0
for i, (x, y) in enumerate(pts):
    print(f"Doing {i}")
    pyautogui.moveTo(x, y, duration=.2, _pause=False)
    time.sleep(.5)

    print("snap")
    ret, frame = cap.read()
    if ret:
        # Save image with timestamp
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_name = f"./data/img_x_{x}_y_{y}_{time_stamp}.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
    time.sleep(.25)


# Xold = np.load("trainig_data_X.npy")
# Yold = np.load("trainig_data_Y.npy")

# # combine old and new data
# X = np.vstack((Xold, X))
# Y = np.vstack((Yold, Y))

# save
np.save("trainig_data_X.npy", X)
np.save("trainig_data_Y.npy", Y)
