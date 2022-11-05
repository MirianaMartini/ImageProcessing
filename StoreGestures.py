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


def startC():
    prompt = 'Press S when Ready --> '
    name = input(prompt)
    if name == 's' or name == 'S':
        global start
        start = True
    else:
        startC()


def StoreGestures(_letter):
    unknownGestureSamples = []
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = htm.handDetector(detectionCon=1)
    i = 0
    averageFlag = True

    # Throw a thread that controls the start of the calculating
    t = Thread(target=startC)
    t.start()

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        RightHand = detector.RightHand(img)  # False = Left Hand;   True = Right Hand
        lmList = detector.findPosition(img, draw=False)
        # print(lmList)

        if len(lmList) != 0 and RightHand is False:  # if a only left hand is detected
            if start is True:
                if i < samples:
                    cv2.putText(img, 'Storing ' + _letter.upper(), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 125), 3,
                                cv2.LINE_AA)
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
            cv2.putText(img, "Remove your Right Hand", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 125), 2,
                        cv2.LINE_AA)

        pTime = fpsShow(img, pTime)  # Show fps number

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    try:
        prompt = 'Which Letter? --> '
        letter = input(prompt)
        StoreGestures(letter)
    except KeyboardInterrupt:
        exit(0)
