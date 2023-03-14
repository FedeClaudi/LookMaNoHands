import cv2
import numpy as np
import time


# Create a SimpleBlobDetector object with the desired parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 30
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByConvexity = True
params.minConvexity = 0.5
params.filterByInertia = True
params.minInertiaRatio = 0.5
detector = cv2.SimpleBlobDetector_create(params)

# Load the HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# Load the input image
image = cv2.imread("test.png")

# Load the Haar Cascade face and eye classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_face(gray_image):
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, 
                                          maxSize=(400, 400), 
                                          minSize=(50, 50)
    )

    if len(faces) == 1:
        x, y, w, h = faces[0]
        return faces[0], gray_image[y:y+h, x:x+w]
    elif len(faces) == 0:
        return None, None
    else:
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        face_image = gray_image[y:y+h, x:x+w]
        return largest_face, face_image
    
def detect_eyes(face_image):
    # Detect eyes in the face image
    eyes = eye_cascade.detectMultiScale(face_image, 
                                        scaleFactor=1.1, 
                                        minNeighbors=10,
                                        maxSize=(100, 100), 
                                        minSize=(50, 50)                      
            )

    # keep the two eyes with the highest y coordinate
    eyes = sorted(eyes, key=lambda eye: eye[1], reverse=True)[:2]
    
    # Return the left and right eye regions
    if len(eyes) >= 2:
        return eyes[0], eyes[1]
    else:
        return None, None

def detect(image):
    
    # Convert the input image to grayscale
    gray_image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    face, face_image = detect_face(gray_image)
    if face is None:
        return
    x, y, w, h = face
    # Draw rectangle around detected face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Detect eyes in the face image
    left, right = detect_eyes(face_image)
    if left is None or right is None:
        return


    # Iterate through each detected eye and crop the eye region
    for (ex, ey, ew, eh) in (left, right):
        cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 2)

        # Crop the eye region from the face region
        eye_img = face_image[ey:ey+eh, ex:ex+ew]

        # Convert the eye region to grayscale
        # eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

        # Detect the position of the pupil in the eye region using SimpleBlobDetector
        # keypoints = detector.detect(eye_img)

        # # If the pupil is detected, draw a circle on the main image around the pupil
        # if len(keypoints) > 0:
        #     pupil_x = int(keypoints[0].pt[0])
        #     pupil_y = int(keypoints[0].pt[1])
        #     cv2.circle(image, (x+ex+pupil_x, y+ey+pupil_y), 5, (0, 255, 0), 2)




    
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)


    # Call the detect function to detect faces and eyes in the frame
    detect(frame)

    # Calculate the FPS and display it on the output image
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the output image
    cv2.imshow("Video", frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
