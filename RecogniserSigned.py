import cv2
import time
import numpy as np
import os
import json
import HandTrackingModule as htm
import FingersUpDownDetector as fUDd
import matplotlib.pyplot as plt
import matplotlib as mpl

tipIds = [4, 8, 12, 16, 20]
keyPoints = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20, 2, 6, 10, 14, 18]
tol = 25
tolIKOP = 40


def find_distances(lmList):
    # calculates, for each node, its signed distance along x and y with all the 21 nodes (with itself too and it's 0)
    dist_matrix_x = np.zeros([len(lmList), len(lmList)], dtype='float')
    dist_matrix_y = np.zeros([len(lmList), len(lmList)], dtype='float')
    palmSize = ((lmList[0][1] - lmList[9][1]) ** 2 + (lmList[0][2] - lmList[9][2]) ** 2) ** (1. / 2.)

    for row in range(0, len(lmList)):
        for column in range(0, len(lmList)):
            dist_matrix_x[row][column] = (lmList[row][1] - lmList[column][1])/palmSize
            dist_matrix_y[row][column] = (lmList[row][2] - lmList[column][2])/palmSize

    distMatrix = [dist_matrix_x, dist_matrix_y]
    return distMatrix


def find_gesture_2(unknownGesture, knownGestures, keyPoints, gestNames, tol):
    # unknown gesture: gesture detected from webcam
    # knownGestures: array of all the gesture for each letter
    # keyPoints: all the key id of the hand
    # gestNames: array of all the Letters Names
    # tol: constant for error
    gesture = 'Unknown'

    errorArray = []
    # For each gestName he finds the error between the gesture related to that Name and
    # the real time detected gesture
    for i in range(0, len(gestNames), 1):
        error = find_error(knownGestures[i], unknownGesture, keyPoints)
        errorArray.append(error)
    errorMin = errorArray[0]
    minIndex = 0

    # finds the min in error array
    for i in range(0, len(errorArray), 1):
        if errorArray[i][0] < errorMin[0] and errorArray[i][1] < errorMin[1]:
            errorMin = errorArray[i]
            minIndex = i
    if errorMin[0] <= tol and errorMin[1] <= tol:
        gesture = gestNames[minIndex]
    if errorMin[0] >= tol and errorMin[1] >= tol:
        gesture = 'Unknown'
    return gesture


def find_gesture(unknownGesture, knownGestures, keyPoints, gestNames, tol):
    # unknown gesture: gesture detected from webcam
    # knownGestures: array of all the gesture for each letter
    # keyPoints: all the key id of the hand
    # gestNames: array of all the Letters Names
    # tol: constant for error
    # tolIKOP: constant for error for letters I, K, O, P
    gesture = 'Unknown'

    errorArray = []
    # For each gestName he finds the error between the gesture related to that Name and
    # the real time detected gesture
    for i in range(0, len(gestNames), 1):
        error = find_error(knownGestures[i], unknownGesture, keyPoints)
        errorArray.append(error)
    errorMin = errorArray[0]
    minIndex = 0

    # finds the min in error array
    for i in range(0, len(errorArray), 1):
        if errorArray[i] < errorMin:
            errorMin = errorArray[i]
            minIndex = i
    if errorMin[0] < tolIKOP and errorMin[1] < tolIKOP:
        if gestNames[minIndex] == 'I' or gestNames[minIndex] == 'K' or gestNames[minIndex] == 'O' \
                or gestNames[minIndex] == 'P':
            gesture = gestNames[minIndex]
            # print(gestNames[minIndex])
        elif errorMin[0] < tol and errorMin[1] < tol:
            gesture = gestNames[minIndex]
        elif errorMin[0] >= tol and errorMin[1] >= tol:
            gesture = 'Unknown'
    return gesture


def find_error(gestureMatrix, unknownMatrix, keyPoints):
    errorX = 0
    errorY = 0

    for row in keyPoints:
        for column in keyPoints:
            errorX = errorX + abs(gestureMatrix[0][row][column] - unknownMatrix[0][row][column])
            errorY = errorY + abs(gestureMatrix[1][row][column] - unknownMatrix[1][row][column])

    error = [errorX, errorY]
    return error


def fps_show(img, pTime):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (500, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    return img, cTime


def get_letter(fileName):
    x = fileName.split('.')
    return x[0]


def get_known_gestures(path):
    gesturesList = []
    namesList = []
    for filename in os.listdir(path):
        with open(str(path+filename), 'r') as outfile:
            gesture = outfile.read()
            gesture = json.loads(gesture)
            gesturesList.append(gesture)

            namesList.append(get_letter(filename))
            outfile.close()
    return namesList, gesturesList  # np.array(gesturesList)


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


def grab_frame(cap, detector, gestNames, knownGestures, pTime):
    """
    Method to grab a frame from the camera and elaborate it in order to detect the hand and signs
    :param cap: the VideoCapture object
    :return: the captured image
    """
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    RightHand = detector.right_hand(img)  # False = Left Hand; True = Right Hand
    lmList = detector.find_position(img, draw=False)

    if len(lmList) != 0 and RightHand is False:  # if a hand is detected
        unknownGesture = find_distances(lmList)
        myGesture = find_gesture(unknownGesture, knownGestures, keyPoints, gestNames, tol)
        fingers_up, fingers_names = fUDd.find_fingers_up(lmList)
        orientation = detector.orientation()
        if orientation is True:
            text = myGesture
            # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            # cv2.putText(img, text, (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
            cv2.putText(img, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 125), 3, cv2.LINE_AA)
            cv2.putText(img, fingers_names, (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 144, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Place your hand correctly", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 125), 2, cv2.LINE_AA)

    if RightHand is True:
        cv2.putText(img, "Remove your Right Hand", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 125), 2, cv2.LINE_AA)

    # img, pTime = fps_show(img, pTime)  # Show fps number

    return img, pTime


def recogniser_signed():
    pTime = 0

    # init the camera
    cap = cv2.VideoCapture(0)

    # enable Matplotlib interactive mode
    plt.ion()

    mpl.use('TkAgg')
    # create a figure to be updated
    fig = plt.figure()
    # intercept the window's close event to call the handle_close() function
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))

    # prep a variable for the first run
    img = None

    # prep detector + load known gestures
    detector = htm.HandDetector()
    gestNames, knownGestures = get_known_gestures("GesturesFilesSigned/")

    while cap.isOpened():
        # get the current frame
        frame, pTime = grab_frame(cap, detector, gestNames, knownGestures, pTime)
        # frame, pTime = grab_frame(cap, detector, gestNames, knownGestures, pTime)

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
        recogniser_signed()
    except KeyboardInterrupt:
        exit(0)