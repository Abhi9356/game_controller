import cv2
import mediapipe as mp
import pyautogui


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


screen_width, screen_height = pyautogui.size()


cam = cv2.VideoCapture(0)
print("Press 'q' to exit.")

def fingers(landmarks) -> int | bool:

    if not landmarks:
        return 0


    thumb_is_open = landmarks[4][0] > landmarks[3][0]


    fingers = []
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]

    for tip, pip in zip(tips, pip_joints):
        fingers.append(landmarks[tip][1] < landmarks[pip][1])

    return thumb_is_open + sum(fingers)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break


    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            landmarks = []
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                landmarks.append((int(lm.x * w), int(lm.y * h)))


            num_fingers = fingers(landmarks)


            index_x, index_y = landmarks[8]


            screen_x = int(screen_width * (index_x / frame.shape[1]))
            screen_y = int(screen_height * (index_y / frame.shape[0]))

            # Gesture Actions
            if num_fingers == 5:
                pyautogui.keyDown("right")
                pyautogui.keyUp("left")
                cv2.putText(frame, "Accelerating", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif num_fingers == 0:
                pyautogui.keyDown("left")
                pyautogui.keyUp("right")
                cv2.putText(frame, "Braking", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif num_fingers == 1:
                pyautogui.moveTo(screen_x, screen_y, duration=0.1)
                cv2.putText(frame, "Cursor Move", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            elif num_fingers == 2:
                pyautogui.click()
                cv2.putText(frame, "Click", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)




    cv2.imshow("Gesture Control", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
