
"""
Use opencv to stream video from a webcam and display + show FPS
"""
import cv2
import time
import numpy as np
from pynput import mouse

# Set up video capture from default camera
cap = cv2.VideoCapture(0)

# set video fps to 60
cap.set(cv2.CAP_PROP_FPS, 60)

# Set up FPS counter
start_time = time.time()

# setup video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

out = cv2.VideoWriter(f'./data/raw/{timestamp}_video.avi', fourcc, 60.0, (640, 480))

# stream
cursor_pos = []
print("Starting")
print("="*250)
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # update every 1s
    if (time.time() - start_time) % 1 < 0.01:
        print(f"Elapsed: {time.time() - start_time:.2f} s")
   
    # if > 10s elapsed, save frame
    if time.time() - start_time > 10:
        out.write(frame)
        
        # store cursor position
        cursor_pos.append(mouse.Controller().position)

    # if 70s elapsed, break
    if time.time() - start_time > 70:
        break

print("Finished, saving")
print("="*250)

# cleanuop
cap.release()
cv2.destroyAllWindows()

# save cursor positions
np.save(f"./data/raw/{timestamp}_cursor_pos.npy", cursor_pos)

# save video
out.release()
