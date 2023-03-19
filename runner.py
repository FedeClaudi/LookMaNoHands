import cv2
import vision
import control


def main():
    # create window centered on the screen
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.moveWindow('frame', 0, 500)

    tracker = vision.Tracker()
    controller = control.Controller()
    while True:
        frame = tracker()
        controller.apply_controls(tracker)

        cv2.imshow('frame', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    tracker.cap.release()


if __name__ == '__main__':
    main()
