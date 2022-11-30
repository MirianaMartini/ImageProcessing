import cv2
import time
import HandTrackingModule as htm
import matplotlib.pyplot as plt

tipIds = [4, 8, 12, 16, 20]
names = ["thumb", "index", "middle", "ring", "picky"]
keyPoints = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]


def fps_show(img, pTime):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (500, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    return img, cTime


def bgr_to_rgb(image):
    """
    Convert a BGR image into RBG
    :param image: the BGR image
    :return: the same image but in RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def handle_close(event, cap):
    """
    Handle the close event of the Matplotlib window by closing the camera capture
    :param event: the close event
    :param cap: the VideoCapture object to be closed
    """
    cap.release()


def find_fingers_up(lmList):
    """
    Finds all the fingers up and returns a list of the names of the fingers up
    :param lmlist: list of coordinates of each node
    """
    fingers_up = []  # [thumb, index, middle, ring, pinky] booleani
    for i in tipIds:
        if i == 4:
            if lmList[i][1] > lmList[i-1][1]:
                fingers_up.append(True)
            else:
                fingers_up.append(False)
        else:
            if lmList[i][2] < lmList[i-1][2]:
                fingers_up.append(True)
            else:
                fingers_up.append(False)

    names_up = ""
    for i in range(len(fingers_up)):
        if fingers_up[i] is True:
            names_up += names[i]
            names_up += ' '

    return fingers_up, names_up


def grab_frame(cap, detector, pTime):
    """
    Method to grab a frame from the camera and elaborate it in order to detect the hand
    :param cap: the VideoCapture object
    :return: the captured image
    """
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    RightHand = detector.RightHand(img)  # False = Left Hand; True = Right Hand
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0 and RightHand is False:  # if a hand is detected
        fingers_up, text = find_fingers_up(lmList)
        # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        # cv2.putText(img, text, (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
        cv2.putText(img, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 125), 3, cv2.LINE_AA)

    if RightHand is True:
        cv2.putText(img, "Remove your Right Hand", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 125), 2, cv2.LINE_AA)

    # img, pTime = fps_show(img, pTime)  # Show fps number

    return img, pTime

def main():
    pTime = 0

    # init the camera
    cap = cv2.VideoCapture(0)

    # enable Matplotlib interactive mode
    plt.ion()

    # create a figure to be updated
    fig = plt.figure()

    # intercept the window's close event to call the handle_close() function
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))

    # prep a variable for the first run
    img = None

    # prep detector + load known gestures
    detector = htm.handDetector(detectionCon=1)

    while cap.isOpened():
        # get the current frame
        frame, pTime = grab_frame(cap, detector, pTime)

        if img is None:
            # convert it in RBG (for Matplotlib)
            img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Recogniser")
            # show the plot!
            plt.show()
        else:
            # set the current frame as the data to show
            img.set_data(bgr_to_rgb(frame))
            # update the figure associated to the shown plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(1 / 30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
