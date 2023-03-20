import cv2
import vision



def main():
    # create window centered on the screen
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.moveWindow('frame', 0, 0)

    tracker = vision.Tracker()
    while True:
        frame = tracker()

        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    tracker.cap.release()


if __name__ == '__main__':
    main()
