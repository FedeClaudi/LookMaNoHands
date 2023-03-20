import cv2
import vision
import control


def main():
    # create window centered on the screen
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.moveWindow('frame', 100, 0)

    tracker = vision.Tracker()
    controller = control.Controller()
    while True:
        frame = tracker()
        controller(tracker)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    tracker.cap.release()


if __name__ == '__main__':
    main()
