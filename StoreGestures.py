import cv2
import time
import numpy as np
from threading import Thread
import HandTrackingModule as htm
import json
import os
import matplotlib.pyplot as plt

samples = 100
start = False
path = "GesturesFiles/"


def findDistances(lmList):  # calculates, for each node, its distance with all the 21 nodes (with itself too and it's 0)
    distMatrix = np.zeros([len(lmList), len(lmList)], dtype='float')
    palmSize = ((lmList[0][1] - lmList[9][1]) ** 2 + (lmList[0][2] - lmList[9][2]) ** 2) ** (1. / 2.)
    for row in range(0, len(lmList)):
        for column in range(0, len(lmList)):
            distMatrix[row][column] = (((lmList[row][1] - lmList[column][1]) ** 2 + (
                        lmList[row][2] - lmList[column][2]) ** 2) ** (1. / 2.)) / palmSize
    return distMatrix


def CalculateAverage(samplesList):
    if len(samplesList) > 0:
        sumG = samplesList[0]
        for i in range(1, len(samplesList)):
            sumG = np.add(sumG, samplesList[i])
        avg = sumG/len(samplesList)
        print(avg)
        return avg
    else:
        return -1


def saveInFile(letterGestureAVG, letterName):
    print('Saving in json')

    # Serializing json
    json_object = json.dumps(letterGestureAVG.tolist())

    # Writing to sample.json
    with open(path + letterName.upper() + ".json", "w+") as outfile:
        outfile.write(json_object)
        outfile.close()


def fpsShow(img, pTime):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (500, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    return cTime

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


def startC():
    prompt = 'Press S when Ready --> '
    name = input(prompt)
    if name == 's' or name == 'S':
        global start
        start = True
    else:
        startC()


def StoreGestures(_letter):
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

    # prep detector + init unknownGestureSamples list
    detector = htm.handDetector(detectionCon=1)
    unknownGestureSamples = []

    # flags
    i = 0
    averageFlag = True

    # Throw a thread that controls the start of the calculating
    t = Thread(target=startC)
    t.start()

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame = detector.findHands(frame)
        RightHand = detector.RightHand(frame)  # False = Left Hand;   True = Right Hand
        lmList = detector.findPosition(frame, draw=False)
        # print(lmList)

        if len(lmList) != 0 and RightHand is False:  # if a only left hand is detected
            if start is True:
                if i < samples:
                    cv2.putText(frame, 'Storing ' + _letter.upper(), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 125),
                                3, cv2.LINE_AA)
                    unknownGestureSample = findDistances(lmList)  # save the sample
                    unknownGestureSamples.append(unknownGestureSample)  # add the sample to the list of samples
                    i = i + 1
                else:
                    if averageFlag is True:
                        res = CalculateAverage(unknownGestureSamples)  # calculate the average
                        saveInFile(res, letter)  # save the result into a file
                        averageFlag = False
                    else:
                        return

        elif RightHand is True:
            cv2.putText(frame, "Remove your Right Hand", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 125), 2,
                        cv2.LINE_AA)

        # frame, pTime = fpsShow(frame, pTime)  # Show fps number

        if img is None:
            # convert it in RBG (for Matplotlib)
            img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Store Gestures")
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
        prompt = 'Which Letter? --> '
        letter = input(prompt)
        StoreGestures(letter)
    except KeyboardInterrupt:
        exit(0)
