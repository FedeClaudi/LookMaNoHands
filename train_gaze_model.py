import numpy as np
import pyautogui
import time
import cv2
import tensorflow as tf

import vision


LEFT = [474,475, 476, 477, 473, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
RIGHT = [469, 470, 471, 472, 468, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
FACE = [227, 116, 137, 123, 345, 346, 280, 352, 208, 199, 428]

SELECTED = [*LEFT, *RIGHT, *FACE]


N = 2*(len(SELECTED))

SAVE_DATA = False
FIT_MODEL = False
T = 1000
snaps_per_frame = 3

X = np.zeros((snaps_per_frame * T, N))
Y = np.zeros((snaps_per_frame * T, 2))
tracker = vision.Tracker()


# ---------------------------------------------------------------------------- #
#                                 GENERATE DATA                                #
# ---------------------------------------------------------------------------- #
if SAVE_DATA:
    screen_w, screen_h = pyautogui.size()

    # select N random points on the screen
    x_pts = np.random.randint(25, screen_w-25, size=(T-4))
    y_pts = np.random.randint(25, screen_h-25, size=(T-4))
    pts = np.vstack((x_pts, y_pts)).T

    # add 4 corners to points
    corners =  np.array([[50, 50], [50, screen_h - 50], [screen_w - 50, screen_h - 50], [screen_w - 50, 50]])
    pts = np.vstack((pts, corners))

    # center cursor
    pyautogui.moveTo(screen_w / 2, screen_h / 2, duration=0, _pause=False)

    

    # move through each point
    j = 0
    for i, (x, y) in enumerate(pts):
        print(f"Doing {i}")
        pyautogui.moveTo(x, y, duration=.2, _pause=False)
        
        time.sleep(.5)
        print("snap")
        x, y = pyautogui.position()
        for k in range(snaps_per_frame):
            Y[j, :] = np.array([x, y])

            tracker()
            X[j, :] = tracker.mesh_points_normalized[SELECTED].ravel()
            j += 1

        time.sleep(.5)
    np.save("trainig_data_X.npy", X)
    np.save("trainig_data_Y.npy", Y)

else:
    X = np.load("trainig_data_X.npy")
    Y = np.load("trainig_data_Y.npy")

# ---------------------------------------------------------------------------- #
#                                  TRAIN MODEL                                 #
# ---------------------------------------------------------------------------- #



# # Initialize a random forest regressor with 100 trees
# # rf = RandomForestRegressor(n_estimators=100, random_state=42)
# model = SVR()

# # Fit the model to the training data
# model.fit(X, Y)

if FIT_MODEL:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss='mse')

    # Define a function that returns the learning rate based on the epoch
    def lr_schedule(epoch):
        if epoch < 100:
            lr = 0.01
        elif epoch < 500:
            lr = 0.005
        elif epoch < 1250:
            lr = 0.001
        else:
            lr = 0.0001
        return lr

    # Create a callback that applies the learning rate schedule
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    model.fit(X, Y, epochs=3500, batch_size=64, callbacks=[lr_callback])

    # Save the model to a file
    model.save("my_mlp_model.h5")
else:
    # Load the model from a file
    model = tf.keras.models.load_model("my_mlp_model.h5")

# ---------------------------------------------------------------------------- #
#                                EVALUATION MODE                               #
# ---------------------------------------------------------------------------- #
print("evaluation mode")
for i in range(1000):
    frame = tracker()
    x = tracker.mesh_points_normalized[SELECTED].ravel()
    y = model.predict(x.reshape(1, -1), verbose=False)
    pyautogui.moveTo(y[0, 0], y[0, 1], duration=0, _pause=False)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.cap.release()


a = 1
