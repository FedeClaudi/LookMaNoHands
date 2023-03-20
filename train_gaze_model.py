import numpy as np
import pyautogui
import time
import cv2
import tensorflow as tf

import vision

tracker = vision.Tracker()




N = 2*(len(tracker.indices))

SAVE_DATA = False
FIT_MODEL = True
RUN = False

T = 100
snaps_per_frame = 3

X = np.zeros((snaps_per_frame * T, N))
Y = np.zeros((snaps_per_frame * T, 2))


# ---------------------------------------------------------------------------- #
#                                 GENERATE DATA                                #
# ---------------------------------------------------------------------------- #
if SAVE_DATA:
    screen_w, screen_h = pyautogui.size()

    # select N random points on the screen
    # x_pts = np.random.randint(5, screen_w-5, size=(T-4))
    # y_pts = np.random.randint(5, screen_h-5, size=(T-4))
    # pts = np.vstack((x_pts, y_pts)).T

    # add 4 corners to points
    delta = 10
    # corners =  np.array([[delta, delta], [delta, screen_h - delta], [screen_w - delta, screen_h - delta], [screen_w - delta, delta]])
    # pts = np.vstack((pts, corners))

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
    pts = np.tile(corners, (4, 1))

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
        x, y = pyautogui.position()
        for k in range(snaps_per_frame):
            Y[j, :] = np.array([x, y])

            tracker()
            X[j, :] = tracker.mesh_points_normalized[tracker.indices].ravel()
            j += 1

        time.sleep(.5)


    Xold = np.load("trainig_data_X.npy")
    Yold = np.load("trainig_data_Y.npy")

    # combine old and new data
    X = np.vstack((Xold, X))
    Y = np.vstack((Yold, Y))

    # save
    np.save("trainig_data_X.npy", X)
    np.save("trainig_data_Y.npy", Y)

else:
    X = np.load("trainig_data_X.npy")
    Y = np.load("trainig_data_Y.npy")
    print(f"Loaded {X.shape[0]} samples.")

# ---------------------------------------------------------------------------- #
#                                  TRAIN MODEL                                 #
# ---------------------------------------------------------------------------- #


if FIT_MODEL:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
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
            lr = 0.0025
        elif epoch < 5000:
            lr = 0.001
        else:    
            lr = 0.0005

        return lr

    # Create a callback that applies the learning rate schedule
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    model.fit(X, Y, epochs=10_000, batch_size=256, callbacks=[lr_callback], use_multiprocessing=True, workers=4)

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
    if not RUN:
        break
    frame = tracker()
    x = tracker.mesh_points_normalized[tracker.indices].ravel()
    y = model.predict(x.reshape(1, -1), verbose=False)
    pyautogui.moveTo(y[0, 0], y[0, 1], duration=0, _pause=False)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.cap.release()


a = 1
