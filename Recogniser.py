import cv2
import time
import numpy as np
import os
import json
import HandTrackingModule as htm
import matplotlib.pyplot as plt

tipIds = [4, 8, 12, 16, 20]
keyPoints = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
tol = 10
#tol = 5

# ha forti difficoltà a riconoscere: m, n,
# si confonde tra: r, u e v / t, x / c_circonflesso, o, p (quest'ultimi più raramente)



def findDistances(lmList): # calculates, for each node, its distance with all the 21 nodes (with itself too and it's 0)
    distMatrix = np.zeros([len(lmList), len(lmList)], dtype='float')
    palmSize = ((lmList[0][1]-lmList[9][1])**2+(lmList[0][2]-lmList[9][2])**2)**(1./2.)
    for row in range(0, len(lmList)):
        for column in range(0, len(lmList)):
            distMatrix[row][column] = (((lmList[row][1]-lmList[column][1])**2+(lmList[row][2]-lmList[column][2])**2)**(1./2.))/palmSize
    return distMatrix


def findGesture(unknownGesture, knownGestures, keyPoints, gestNames, tol):
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
        error = findError(knownGestures[i], unknownGesture, keyPoints)
        errorArray.append(error)
    errorMin = errorArray[0]
    minIndex = 0

    # finds the min in error array
    for i in range(0, len(errorArray), 1):
        if errorArray[i] < errorMin:
            errorMin = errorArray[i]
            minIndex = i
    if errorMin < tol:
        gesture = gestNames[minIndex]
    if errorMin >= tol:
        gesture = 'Unknown'
    return gesture


def findError(gestureMatrix, unknownMatrix, keyPoints):
    error = 0
    for row in keyPoints:
        for column in keyPoints:
            error = error + abs(gestureMatrix[row][column] - unknownMatrix[row][column])
    return error


def fpsShow(img, pTime):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (500, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    return cTime


def getLetter(fileName):
    x = fileName.split('.')
    return x[0]


def getKnownGestures(path):
    gesturesList = []
    namesList = []
    for filename in os.listdir(path):
        with open(str(path+filename), 'r') as outfile:
            gesture = outfile.read()
            gesture = json.loads(gesture)
            gesturesList.append(gesture)

            namesList.append(getLetter(filename))
            outfile.close()
    return namesList, gesturesList  # np.array(gesturesList)


def main():
    wCam, hCam = 640, 480

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    pTime = 0

    detector = htm.handDetector(detectionCon=1)

    gestNames, knownGestures = getKnownGestures("GesturesFiles/")

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        RightHand = detector.RightHand(img)  # False = Left Hand; True = Right Hand
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0 and RightHand is False:  # if a hand is detected
            unknownGesture = findDistances(lmList)
            myGesture = findGesture(unknownGesture, knownGestures, keyPoints, gestNames, tol)
            text = myGesture
            # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            # cv2.putText(img, text, (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
            cv2.putText(img, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 125), 3, cv2.LINE_AA)

        if RightHand is True:
            cv2.putText(img, "Remove your Right Hand", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 125), 2,
                        cv2.LINE_AA)

        pTime = fpsShow(img, pTime) #Show fps number

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
