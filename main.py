import cv2
import mediapipe as mp

# Access solutions safely across MediaPipe versions
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# Setup Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def is_spiderman_gesture(hand_landmarks):
    """
    Detects classic Spider-Man hand gesture:
    - Index (8) & Pinky (20) extended.
    - Middle (12) & Ring (16) folded down.
    """
    # Tip landmarks
    index_tip = hand_landmarks.landmark[8].y
    middle_tip = hand_landmarks.landmark[12].y
    ring_tip = hand_landmarks.landmark[16].y
    pinky_tip = hand_landmarks.landmark[20].y

    # Knuckle/PIP joint landmarks (to compare fold state)
    index_pip = hand_landmarks.landmark[6].y
    middle_pip = hand_landmarks.landmark[10].y
    ring_pip = hand_landmarks.landmark[14].y
    pinky_pip = hand_landmarks.landmark[18].y

    # In image coordinates, smaller Y means HIGHER up on screen
    index_open = index_tip < index_pip
    pinky_open = pinky_tip < pinky_pip
    middle_closed = middle_tip > middle_pip
    ring_closed = ring_tip > ring_pip

    return index_open and pinky_open and middle_closed and ring_closed

# Start webcam capture
cap = cv2.VideoCapture(0)

print("Starting Spider-Man Web Shooter... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # Check for Spider-Man gesture
            if is_spiderman_gesture(hand_landmarks):
                # Get wrist point for web anchor (Landmark 0)
                wrist_x = int(hand_landmarks.landmark[0].x * w)
                wrist_y = int(hand_landmarks.landmark[0].y * h)

                # Visual indicator on palm/wrist
                cv2.circle(frame, (wrist_x, wrist_y), 15, (0, 255, 255), -1)
                cv2.putText(frame, "THWIP! WEB SHOOTING!", (wrist_x - 100, wrist_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Spider-Man Web Shooter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()