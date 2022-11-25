import cv2
from threading import Thread
import HandTrackingModule as htm
import matplotlib.pyplot as plt
import matplotlib as mpl
import RecogniserSigned as Recogniser

#test_path = "TestSigned/TestRoberta.txt"
test_path = "TestSigned/TestMiriana.txt"

samples = 100
known_gestures = []
gest_names = []
start = False


def update_file(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write("{}".format(line))
        f.close()


def save_in_file(path, _letter, value):
    flag = True
    lines = []

    try:
        with open(path, "r+") as f:
            i = 0

            lines = f.readlines()
            f.close()

            for line in lines:
                x = line.split(':')
                if x[0] == _letter:
                    new_line = "{}: {}/{}\n".format(_letter, value, samples)
                    lines[i] = new_line
                    flag = False
                i += 1
        if flag:
            new_line = "{}: {}/{}\n".format(_letter, value, samples)
            lines.append(new_line)

        update_file(path, lines)

    except IOError:
        new_line = "{}: {}/{}\n".format(_letter, value, samples)
        lines.append(new_line)
        update_file(path, lines)


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


def start_c():
    prompt = 'Press S when Ready --> '
    name = input(prompt)
    if name == 's' or name == 'S':
        global start
        start = True
    else:
        start_c()


def exist(_letter):
    global known_gestures, gest_names
    gest_names, known_gestures = Recogniser.get_known_gestures(Recogniser.path_m)
    for name in gest_names:
        if name == _letter:
            return True
    return False


def test(_letter):
    pTime = 0

    mpl.use('TkAgg')

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

    # prep detector + init unknownGestureSamples list
    detector = htm.HandDetector()
    unknown_gesture_samples = []

    # flags
    i = 0
    averageFlag = True

    # Throw a thread that controls the start of the calculating
    t = Thread(target=start_c)
    t.start()

    true_positive = 0

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame = detector.find_hands(frame)
        RightHand = detector.right_hand(frame)  # False = Left Hand;   True = Right Hand
        lmList = detector.find_position(frame, draw=False)

        if len(lmList) != 0 and RightHand is False:  # if a only left hand is detected
            orientation = detector.orientation()
            if orientation is True:
                unknown_gesture_sample = Recogniser.find_distances(lmList)  # save the sample
                global known_gestures, gest_names
                myGesture = Recogniser.find_gesture(unknown_gesture_sample, known_gestures, Recogniser.keyPoints,
                                                    gest_names)
                cv2.putText(frame, myGesture, (2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 190, 1), 3, cv2.LINE_AA)

                if start is True:
                    if i < samples:
                        cv2.putText(frame, 'Testing ' + _letter.upper(), (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (0, 0, 125), 3, cv2.LINE_AA)
                        if myGesture == _letter:
                            true_positive += 1
                        i = i + 1
                    else:
                        print("Correct Guess: {}/{}".format(true_positive, samples))
                        save_in_file(test_path, _letter, true_positive)
                        return
                else:
                    cv2.putText(frame, 'Press S to start Testing', (2, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 125),
                                3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Place your hand correctly", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 125), 2,
                            cv2.LINE_AA)

        elif RightHand is True:
            cv2.putText(frame, "Remove your Right Hand", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 125), 2,
                        cv2.LINE_AA)

        # frame, pTime = fps_show(frame, pTime)  # Show fps number

        if img is None:
            # convert it in RBG (for Matplotlib)
            img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Test Signed")
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
    ok = True
    try:
        while ok:
            prompt = 'Indicate the Letter to be tested! --> '
            letter = input(prompt)
            if exist(letter.upper()):
                test(letter.upper())
                ok = False
            else:
                ok = True
    except KeyboardInterrupt:
        exit(0)
