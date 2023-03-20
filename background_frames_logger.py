import cv2
from datetime import datetime
from pynput import mouse
import time

# Set up video capture from default camera
cap = cv2.VideoCapture(0)

# Define image counter variable
img_counter = 0

def on_click(x, y, button, pressed):
    if not pressed:
        return
    
    # Capture frame from webcam
    x, y = int(x), int(y)
    ret, frame = cap.read()
    if ret:
        # Save image with timestamp
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_name = f"./data/img_x_{x}_y_{y}_{time_stamp}.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))


listener = mouse.Listener(on_click=on_click)
listener.start()
listener.join()

while True:
    time.sleep(.1)


# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()