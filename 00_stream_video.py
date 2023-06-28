
"""
Use opencv to stream video from a webcam and display + show FPS
"""
import cv2
import time
import numpy as np

"""
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


"""

# Set up video capture from default camera
cap = cv2.VideoCapture(0)

# set video fps to 60
cap.set(cv2.CAP_PROP_FPS, 60)

# Set up FPS counter
frames = 0
start_time = time.time()


# stream
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if ret:
        # Display frame
        frames += 1

        # Calculate FPS
        if frames > 10:
            fps = round(frames / (time.time() - start_time), 2)
        else:
            fps = 0

        # draw landmarks

        # write fps on frame
        frame = cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("frame", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanuop
cap.release()
cv2.destroyAllWindows()