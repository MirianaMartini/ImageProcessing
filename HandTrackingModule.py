"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""
import cv2
import mediapipe as mp
import numpy as np
import time


class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks: #se ha trovato una mano
            for handLms in self.results.multi_hand_landmarks:
                for idx, classification in enumerate(self.results.multi_handedness):
                    if classification.classification[0].label == 'Left': #Se ha individuato la mano Sinistra
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def RightHand(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        error = None

        if self.results.multi_handedness:  # se ha trovato una mano
            for idx, classification in enumerate(self.results.multi_handedness):
                if classification.classification[0].label == 'Right':  # Mando Destra --> errore
                    error = True
                if classification.classification[0].label == 'Left':  # Mano Sinistra --> ok
                    error = False
        return error

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList

    def orientation(self, img):
        if self.results.multi_handedness:  # se ha trovato una mano
            flag = False
            for hand in self.results.multi_hand_landmarks:
                if hand.landmark[0].y > hand.landmark[1].y and hand.landmark[0].y > hand.landmark[17].y:
                    flag = True
                else:
                    flag = False
                """"
                elif hand.landmark[0].y < hand.landmark[1].y and hand.landmark[0].y < hand.landmark[17].y:
                    flag = False
                    print("mano rivolta verso il basso")
                elif hand.landmark[17].y < hand.landmark[0].y and hand.landmark[1].y > hand.landmark[0].y:
                    flag = False
                    print("mano rivolta verso destra")
                elif hand.landmark[17].y > hand.landmark[0].y and hand.landmark[1].y < hand.landmark[0].y:
                    flag = False
                    print("mano rivolta verso sinistra")
                """
            return flag



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
