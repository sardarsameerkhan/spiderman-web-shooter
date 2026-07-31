import cv2
import mediapipe as mp
import numpy as np
import math

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
    """Detects classic Spider-Man hand gesture."""
    index_tip = hand_landmarks.landmark[8].y
    middle_tip = hand_landmarks.landmark[12].y
    ring_tip = hand_landmarks.landmark[16].y
    pinky_tip = hand_landmarks.landmark[20].y

    index_pip = hand_landmarks.landmark[6].y
    middle_pip = hand_landmarks.landmark[10].y
    ring_pip = hand_landmarks.landmark[14].y
    pinky_pip = hand_landmarks.landmark[18].y

    index_open = index_tip < index_pip
    pinky_open = pinky_tip < pinky_pip
    middle_closed = middle_tip > middle_pip
    ring_closed = ring_tip > ring_pip

    return index_open and pinky_open and middle_closed and ring_closed

def draw_spider_web(img, start_point, target_point, progress=1.0):
    """
    Draws a realistic animated spider web stream from start_point towards target_point.
    """
    sx, sy = start_point
    tx, ty = target_point

    # Calculate current end point based on animation progress
    curr_tx = int(sx + (tx - sx) * progress)
    curr_ty = int(sy + (ty - sy) * progress)

    # Main web core (bright white center line)
    cv2.line(img, (sx, sy), (curr_tx, curr_ty), (255, 255, 255), 4)
    # Web glow/outer line
    cv2.line(img, (sx, sy), (curr_tx, curr_ty), (200, 200, 255), 1)

    # Draw branching web strands
    angle = math.atan2(curr_ty - sy, curr_tx - sx)
    length = math.hypot(curr_tx - sx, curr_ty - sy)

    num_branches = 5
    for i in range(1, num_branches + 1):
        dist = (length / num_branches) * i
        bx = int(sx + dist * math.cos(angle))
        by = int(sy + dist * math.sin(angle))

        # Side web flares
        flare_len = int(15 * (i / num_branches))
        perp_angle1 = angle + math.pi / 3
        perp_angle2 = angle - math.pi / 3

        fx1 = int(bx + flare_len * math.cos(perp_angle1))
        fy1 = int(by + flare_len * math.sin(perp_angle1))
        fx2 = int(bx + flare_len * math.cos(perp_angle2))
        fy2 = int(by + flare_len * math.sin(perp_angle2))

        cv2.line(img, (bx, by), (fx1, fy1), (240, 240, 255), 1)
        cv2.line(img, (bx, by), (fx2, fy2), (240, 240, 255), 1)

    # Web impact circle at target point
    if progress >= 0.9:
        cv2.circle(img, (curr_tx, curr_ty), 12, (255, 255, 255), 2)
        cv2.circle(img, (curr_tx, curr_ty), 5, (0, 255, 255), -1)

# Start webcam capture
cap = cv2.VideoCapture(0)

# Animation tracker dict for active webs: {hand_id: frame_count}
web_animation_frames = {}

print("Starting Spider-Man Web Shooter with Animated Webs! Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    current_hands_shooting = []

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )

            if is_spiderman_gesture(hand_landmarks):
                current_hands_shooting.append(idx)

                # Web source position: Wrist (0)
                wrist_x = int(hand_landmarks.landmark[0].x * w)
                wrist_y = int(hand_landmarks.landmark[0].y * h)

                # Direction calculated from wrist through middle finger base (9)
                mid_x = int(hand_landmarks.landmark[9].x * w)
                mid_y = int(hand_landmarks.landmark[9].y * h)

                # Calculate shooting angle outwards
                dx = mid_x - wrist_x
                dy = mid_y - wrist_y
                norm = math.hypot(dx, dy) + 1e-5
                dir_x = dx / norm
                dir_y = dy / norm

                # Target point far along the gesture direction
                target_x = int(wrist_x + dir_x * 800)
                target_y = int(wrist_y + dir_y * 800)

                # Increment web shoot progress
                anim_frame = web_animation_frames.get(idx, 0) + 1
                web_animation_frames[idx] = anim_frame
                progress = min(1.0, anim_frame / 5.0)  # Reaches full distance in 5 frames

                # Draw web graphic
                draw_spider_web(frame, (wrist_x, wrist_y), (target_x, target_y), progress)

                # "THWIP!" HUD text overlay (Fixed OpenCV font constant)
                cv2.putText(frame, "THWIP!", (wrist_x - 40, wrist_y - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, "THWIP!", (wrist_x - 40, wrist_y - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)

    # Clean up inactive hand animations
    web_animation_frames = {k: v for k, v in web_animation_frames.items() if k in current_hands_shooting}

    cv2.imshow('Spider-Man Web Shooter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()