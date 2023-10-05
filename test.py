
import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hand model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 represents the default webcam

# Get screen dimensions
screen_width, screen_height = 3840, 2160  # Update with your screen resolution
ret, frame = cap.read()

# Get frame dimensions
frame_height, frame_width, _ = frame.shape



# Calculate the dimensions of the rectangle in the top-right corner
offset_x, offset_y, scale = 125, 125, 0.3
rect_width = int(frame_width * scale)
rect_height = int((rect_width / screen_width) * screen_height)



def get_index_fingertip_position(hand_landmarks):
    if hand_landmarks:
        # Get the coordinates of the index fingertip
        index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = index_fingertip.x * frame_width, index_fingertip.y * frame_height

        # Check if the fingertip is inside the rectangle
        if offset_x <= x <= offset_x + rect_width and offset_y <= y <= offset_y + rect_height:
            # Calculate the position within the rectangle in [0, 1] range
            position_x = (x - offset_x) / rect_width
            position_y = (y - offset_y) / rect_height
            # return position_x, position_y
            return (1 - position_x) * screen_width, (1 - position_y) * screen_height

    # Return (-1, -1) if fingertip is outside the rectangle
    return -1, -1


# stream
with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        # Draw the rectangle
        # Draw the rectangle
        cv2.rectangle(frame, (offset_x, offset_y),
                      (offset_x + rect_width, offset_y + rect_height), (0, 255, 0), 2)

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hand model
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x, y = get_index_fingertip_position(hand_landmarks)
                if x < 0 or y < 0: 
                    continue
                pyautogui.moveTo(max(1, x), max(1, y), duration=0.0, _pause=False)


        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
