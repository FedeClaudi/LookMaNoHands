import cv2
import vision


# add face pose: https://github.com/Rassibassi/mediapipeDemos/blob/main/head_posture.py



def main():
    # create window centered on the screen
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.moveWindow('frame', 0, 0)

    tracker = vision.Tracker()
    while True:
        frame = tracker(control=True)
        cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
    tracker.cap.release()


if __name__ == '__main__':
    main()
