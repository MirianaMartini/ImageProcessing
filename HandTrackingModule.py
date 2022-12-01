import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, max_hands=1, model_complex=1, detection_con=0.5, track_con=0.6):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complex = model_complex
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complex, self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = 0

    def find_hands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for idx, classification in enumerate(self.results.multi_handedness):
                    if classification.classification[0].label == 'Left':
                        self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def right_hand(self, img):
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

    def find_position(self, img, hand_no=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for idn, lm in enumerate(myHand.landmark):
                # print(idn, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(idn, cx, cy)
                lmList.append([idn, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

    """
    def get_coordinates(self):
        hands = []
        if self.results.multi_handedness:  # se ha trovato una mano
            for hand in self.results.multi_hand_landmarks:
                hands.append(hand)
        if len(hands) != 0:
            return hands[0]
        else:
            return 0
    """

    def orientation(self):
        if self.results.multi_handedness:  # se ha trovato una mano
            flag = False
            for hand in self.results.multi_hand_landmarks:
                cos_right = (hand.landmark[17].x-hand.landmark[0].x)**2/((hand.landmark[17].x-hand.landmark[0].x)**2
                                                                         + (hand.landmark[17].y-hand.landmark[0].y)**2)
                cos_left = (hand.landmark[0].x - hand.landmark[1].x) ** 2/((hand.landmark[0].x-hand.landmark[1].x)**2
                                                                           + (hand.landmark[1].y-hand.landmark[0].y)**2)
                if hand.landmark[0].y > hand.landmark[1].y and hand.landmark[0].y > hand.landmark[17].y:
                    if (0 <= cos_right <= 0.4) and (0 <= cos_left <= 0.9):
                        flag = True
                    else:
                        flag = False
                else:
                    flag = False
            return flag
