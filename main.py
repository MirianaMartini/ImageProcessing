import cv2
import time
import os
import matplotlib.pyplot as plt
import HandTrackingModule as htm

tipIds = [4, 8, 12, 16, 20]

def checkA(lmList):
    print(lmList[tipIds[0]][1] - lmList[tipIds[1] - 2][1])
    if lmList[tipIds[0]][1] > lmList[tipIds[1] - 1][1] and lmList[tipIds[0]][1] - lmList[tipIds[1] - 2][1] <= 24: #thumb
        if lmList[tipIds[1]][2] > lmList[tipIds[1] - 3][2] and lmList[tipIds[1] - 1][2] > lmList[tipIds[1] - 3][2]: #index
            if lmList[tipIds[2]][2] > lmList[tipIds[2] - 3][2] and lmList[tipIds[2] - 1][2] > lmList[tipIds[2] - 3][2]: #middle finger
                if lmList[tipIds[3]][2] > lmList[tipIds[3] - 3][2] and lmList[tipIds[3] - 1][2] > lmList[tipIds[3] - 3][2]: #ring finger
                    if lmList[tipIds[4]][2] > lmList[tipIds[4] - 3][2] and lmList[tipIds[4] - 1][2] > lmList[tipIds[4] - 3][2]:  #pinky
                        return True
    return False

def checkB(lmList):
    if lmList[tipIds[0]][2] < lmList[tipIds[0] - 1][2]: #thumb
        if lmList[tipIds[1]][2] < lmList[tipIds[1] - 2][2]: #index
            if lmList[tipIds[2]][2] < lmList[tipIds[2] - 2][2]: #middle finger
                if lmList[tipIds[3]][2] < lmList[tipIds[3] - 2][2]: #ring finger
                    if lmList[tipIds[4]][2] < lmList[tipIds[4] - 2][2]:  #pinky
                        if lmList[tipIds[1]][1]-lmList[tipIds[2]][1] <= 20:
                            if lmList[tipIds[2]][1] - lmList[tipIds[3]][1] <= 20:
                                if lmList[tipIds[3]][1] - lmList[tipIds[4]][1] <= 24:
                                    if lmList[tipIds[0]][1] - lmList[tipIds[1]][1] <= 15:
                                        return True
    return False

def checkC(lmList):
    if lmList[tipIds[0]][2] < lmList[tipIds[0] - 1][2]:  # thumby
        if lmList[tipIds[0]][1] < lmList[tipIds[1] - 3][1]:  # thumbx
            if lmList[tipIds[1]][2] < lmList[tipIds[1] - 1][2] and lmList[tipIds[1]][2] > lmList[tipIds[1] - 3][2]:  # index
                if lmList[tipIds[2]][2] < lmList[tipIds[2] - 1][2] and lmList[tipIds[2]][2] > lmList[tipIds[2] - 3][2]:  # middle
                    if lmList[tipIds[3]][2] < lmList[tipIds[3] - 1][2] and lmList[tipIds[3]][2] > lmList[tipIds[3] - 3][2]:  # ring
                        if lmList[tipIds[4]][2] < lmList[tipIds[4] - 1][2] and lmList[tipIds[4]][2] > lmList[tipIds[4] - 3][2]:  # pinky
                            if lmList[tipIds[1]][1] - lmList[tipIds[2]][1] <= 20:
                                if lmList[tipIds[2]][1] - lmList[tipIds[3]][1] <= 20:
                                    if lmList[tipIds[3]][1] - lmList[tipIds[4]][1] <= 24:
                                        return True
    return False


def fpsShow(img, pTime):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

def main():
    wCam, hCam = 640, 480

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    pTime = 0

    detector = htm.handDetector(detectionCon=1)

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        # print(lmList)

        if len(lmList) != 0: #if a hand is detected

            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)

            text = "None"

            if checkA(lmList):
                text = "A"
            elif checkB(lmList):
                text = "B"
            elif checkC(lmList):
                text = "C"

            cv2.putText(img, text, (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        fpsShow(img, pTime) #Show fps number

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)