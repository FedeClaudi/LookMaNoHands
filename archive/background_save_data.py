import numpy as np
import glob 
import cv2
import os


import vision

tracker = vision.Tracker()
tracker(control=False)
features = tracker.extract_features()

img_files = list(glob.glob('accepted/*.png'))

X_new = np.zeros((len(img_files), features.shape[0]), dtype=np.float32)
Y_new = np.zeros((len(img_files), 2), dtype=np.float32)


# load all images in data
for i, filename in enumerate(img_files):
    if i % 25 == 0:
        print(f"Doing {i+1}/{len(img_files)}")
    x, y = filename.split("_")[2], filename.split("_")[4]
    x, y = int(float(x)), int(float(y))

    # load image with opencv
    img = cv2.imread(filename)
    img = cv2.flip(img, 1)

    # run tracker on image
    for _ in range(5):
        success = tracker.get_face_mesh(img)
        tracker.get_head_transform()

    if not success:
        print("No face found")
        # delete image
        os.remove(filename)
        continue

    # store data
    X_new[i, :] = tracker.extract_features().astype(np.float32)
    Y_new[i, :] = np.array([x, y]).astype(np.float32)

# # load old data
# X_old = np.load("trainig_data_X.npy")
# Y_old = np.load("trainig_data_Y.npy")

# # merge and save
# X = np.vstack((X_old, X_new))
# Y = np.vstack((Y_old, Y_new))

print(f"X shape: {X_new.shape}")
np.save("trainig_data_X.npy", X_new)
np.save("trainig_data_Y.npy", Y_new)

print("Done, you can delete the images in the data folder now")