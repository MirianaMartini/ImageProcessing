import cv2
import numpy as np
import os
import json
import HandTrackingModule as htm
import matplotlib.pyplot as plt
import matplotlib as mpl

tipIds = [4, 8, 12, 16, 20]
keyPoints = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20, 2, 6, 10, 14, 18]
tol = 25
tol_max = 40
tol_ISE = 50
path_m = "GesturesFilesSigned/"


def find_distances(lmList):
    """
    Calculates, for each node, its signed distance along x and y with all the 21 nodes (with itself too and it's 0)
    :param lmList: matrix of coordinates for each node
    :return: matrix of tuples 21x21 (each tuple contains the distance along x and along y with sign for the verse)
    """
    dist_matrix_x = np.zeros([len(lmList), len(lmList)], dtype='float')
    dist_matrix_y = np.zeros([len(lmList), len(lmList)], dtype='float')
    palmSize = ((lmList[0][1] - lmList[9][1]) ** 2 + (lmList[0][2] - lmList[9][2]) ** 2) ** (1. / 2.)

    for row in range(0, len(lmList)):
        for column in range(0, len(lmList)):
            dist_matrix_x[row][column] = (lmList[row][1] - lmList[column][1])/palmSize
            dist_matrix_y[row][column] = (lmList[row][2] - lmList[column][2])/palmSize

    distMatrix = [dist_matrix_x, dist_matrix_y]
    return distMatrix


def find_gesture(unknownGesture, knownGestures, gestNames):
    """
    Finds the closest gesture to the one detected from the recogniser or return "Unknown"
    :param unknownGesture: matrix of distances between a specific node and all the others related to the detected gesture
    :param knownGestures: matrix of distances between all the 21 nodes
    :param gestNames: array containing all the names of all the letters stored
    :return: gesture found or "Unknown"
    """
    gesture = 'Unknown'

    errorArray = []
    # For each gestName he finds the error between the gesture related to that Name and
    # the real time detected gesture
    for i in range(0, len(gestNames), 1):
        error = find_error(knownGestures[i], unknownGesture)
        errorArray.append(error)
    errorMin = errorArray[0]
    minIndex = 0

    # finds the min in error array
    for i in range(0, len(errorArray), 1):
        if errorArray[i] < errorMin:
            errorMin = errorArray[i]
            minIndex = i
    if errorMin[0] < tol_ISE and errorMin[1] < tol_ISE:
        if gestNames[minIndex] == 'I' or gestNames[minIndex] == 'S' or gestNames[minIndex] == 'E':
            gesture = gestNames[minIndex]
        elif errorMin[0] < tol_max and errorMin[1] < tol_max:
            if gestNames[minIndex] == 'K' or gestNames[minIndex] == 'O' or gestNames[minIndex] == 'P' \
                    or gestNames[minIndex] == 'C_CIRCONFLESSO' \
                    or gestNames[minIndex] == 'G' or gestNames[minIndex] == 'M' or gestNames[minIndex] == 'T' \
                    or gestNames[minIndex] == 'N' or gestNames[minIndex] == 'Q' or gestNames[minIndex] == 'E':
                gesture = gestNames[minIndex]
            elif errorMin[0] < tol and errorMin[1] < tol:
                gesture = gestNames[minIndex]
            elif errorMin[0] >= tol and errorMin[1] >= tol:
                gesture = 'Unknown'
    return gesture


def find_error(gestureMatrix, unknownMatrix):
    """
    Calculates an error between the gesture and the one detected for each coordinate (x, y)
    :param gestureMatrix: matrix of distances between a specific node and all the others related to a specific letter
    :param unknownMatrix: matrix of distances between a specific node and all the others related to the detected gesture
    :return: tuple of the error calculated (Ex, Ey)
    """
    errorX = 0
    errorY = 0

    for row in keyPoints:
        for column in keyPoints:
            errorX = errorX + abs(gestureMatrix[0][row][column] - unknownMatrix[0][row][column])
            errorY = errorY + abs(gestureMatrix[1][row][column] - unknownMatrix[1][row][column])

    error = [errorX, errorY]
    return error


def get_letter(fileName):
    """
    Reads the knows gestures data from files
    :param fileName: name of the file containing a specific letter data
    :return: the name of the letter
    """
    x = fileName.split('.')
    return x[0]


def get_known_gestures(path):
    """
    Reads the knows gestures data from files
    :param path: path of the files to be read
    :return: a names List of all the letters available and a gestures List of the distances
    """
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
    Converts a BGR image into RBG
    :param image: the BGR image
    :return: the same image but in RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def handle_close(event, cap):
    """
    Handles the close event of the Matplotlib window by closing the camera capture
    :param event: the close event
    :param cap: the VideoCapture object to be closed
    """
    cap.release()


def grab_frame(cap, detector, gestNames, knownGestures):
    """
    Method to grab a frame from the camera and elaborate it in order to detect the hand and signs
    :param cap: the VideoCapture object
    :param detector: HandDetector object
    :param gestNames: array containing all the names of all the letters stored
    :param knownGestures: array of 21 arrays (one for each letter containing the 21 distances)
    :return: the captured image
    """
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    RightHand = detector.right_hand(img)  # False = Left Hand; True = Right Hand
    lmList = detector.find_position(img, draw=False)

    if len(lmList) != 0 and RightHand is False:  # if a hand is detected
        unknownGesture = find_distances(lmList)
        myGesture = find_gesture(unknownGesture, knownGestures, gestNames)
        # fingers_up, fingers_names = fUDd.find_fingers_up(lmList)
        orientation = detector.orientation()
        if orientation is True:
            text = myGesture
            cv2.putText(img, text, (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 125), 3, cv2.LINE_AA)
            # cv2.putText(img, fingers_names, (2, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 144, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Place your hand correctly", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 125), 2,
                        cv2.LINE_AA)

    if RightHand is True:
        cv2.putText(img, "Remove your Right Hand", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 125), 2, cv2.LINE_AA)

    # img, pTime = fps_show(img, pTime)  # Show fps number

    return img


def recogniser_signed():
    """
    Method to start a script able to recognise the letters of the sign language
    """
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
    gestNames, knownGestures = get_known_gestures(path_m)

    while cap.isOpened():
        # get the current frame
        frame = grab_frame(cap, detector, gestNames, knownGestures)
        # frame, pTime = grab_frame(cap, detector, gestNames, knownGestures, pTime)

        if img is None:
            # convert it in RBG (for Matplotlib)
            img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Recogniser Signed")

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
