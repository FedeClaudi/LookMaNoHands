import numpy as np
import pyautogui
import time
import cv2
import tensorflow as tf

import vision

tracker = vision.Tracker()
tracker(control=False)
features = tracker.extract_features() 

N_iters = 30_000


# ---------------------------------------------------------------------------- #
#                                 GENERATE DATA                                #
# ---------------------------------------------------------------------------- #
X = np.load("trainig_data_X.npy")
Y = np.load("trainig_data_Y.npy")
print(f"\n\n\n\nLoaded {X.shape[0]} samples.\n\n\n\n")


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
    if epoch < 500:
        lr = 0.01
    elif epoch < 2000:
        lr = 0.005
    elif epoch < 4000:
        lr = 0.0025
    elif epoch < 8000:
        lr = 0.001
    elif epoch < 15000:
        lr = 0.0005
    else:    
        lr = 0.0001

    return lr

# Create a callback that applies the learning rate schedule
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

model.fit(X, Y, epochs=N_iters, batch_size=256, callbacks=[lr_callback], )
        #   use_multiprocessing=True, workers=4)

# Save the model to a file
model.save("my_mlp_model.h5")
