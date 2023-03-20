import numpy as np
import glob 
import cv2


import vision

tracker = vision.Tracker()

img_files = list(glob.glob('data/*.png'))

X_new = np.zeros((len(img_files), 2 * len(tracker.indices)))
Y_new = np.zeros((len(img_files), 2))


# load all images in data
for i, filename in enumerate(img_files):
    x, y = filename.split("_")[2], filename.split("_")[4]

    # load image with opencv
    img = cv2.imread(filename)
    img = cv2.flip(img, 1)

    # run tracker on image
    tracker.get_face_mesh(img)


    # store data
    X_new[i, :] = tracker.mesh_points_normalized[tracker.indices].ravel()
    Y_new[i, :] = np.array([x, y])

# load old data
X_old = np.load("trainig_data_X.npy")
Y_old = np.load("trainig_data_Y.npy")

# merge and save
X = np.vstack((X_old, X_new))
Y = np.vstack((Y_old, Y_new))

print("Done, you can delete the images in the data folder now")