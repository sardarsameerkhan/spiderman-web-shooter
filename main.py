import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random

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
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# Active webs stuck on screen
stuck_screen_webs = []

def is_open_palm(hand_landmarks):
    """Detects fully open palm."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    open_fingers = [hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y for tip, pip in zip(tips, pips)]
    return all(open_fingers)

def is_spiderman_gesture(hand_landmarks):
    """Detects Spider-Man 'Thwip' gesture (Index + Pinky extended, Middle + Ring down)."""
    index_tip = hand_landmarks.landmark[8].y
    middle_tip = hand_landmarks.landmark[12].y
    ring_tip = hand_landmarks.landmark[16].y
    pinky_tip = hand_landmarks.landmark[20].y

    index_pip = hand_landmarks.landmark[6].y
    middle_pip = hand_landmarks.landmark[10].y
    ring_pip = hand_landmarks.landmark[14].y
    pinky_pip = hand_landmarks.landmark[18].y

    return (index_tip < index_pip) and (pinky_tip < pinky_pip) and (middle_tip > middle_pip) and (ring_tip > ring_pip)

def draw_hand_palm_web(img, center, radius):
    """Draws a complete realistic cobweb matrix over the open palm."""
    cx, cy = center
    num_spokes = 8
    rings = 4

    # Outer glow ring
    cv2.circle(img, (cx, cy), int(radius * 1.1), (255, 255, 255), 1)

    # Spokes
    for i in range(num_spokes):
        angle = i * (2 * math.pi / num_spokes)
        sx = int(cx + radius * math.cos(angle))
        sy = int(cy + radius * math.sin(angle))
        cv2.line(img, (cx, cy), (sx, sy), (240, 240, 255), 2)

    # Concentric cobweb rings
    for r in range(1, rings + 1):
        r_dist = int((radius / rings) * r)
        ring_pts = []
        for i in range(num_spokes):
            angle = i * (2 * math.pi / num_spokes)
            rx = int(cx + r_dist * math.cos(angle))
            ry = int(cy + r_dist * math.sin(angle))
            ring_pts.append((rx, ry))

        for i in range(num_spokes):
            pt1 = ring_pts[i]
            pt2 = ring_pts[(i + 1) % num_spokes]
            cv2.line(img, pt1, pt2, (220, 220, 255), 1)

def create_screen_splat(x, y):
    """Generates a realistic stuck web splat pattern."""
    num_strands = random.randint(10, 14)
    radius = random.randint(130, 190)
    strands = []

    for _ in range(num_strands):
        angle = random.uniform(0, 2 * math.pi)
        length = random.uniform(radius * 0.5, radius)
        ex = int(x + length * math.cos(angle))
        ey = int(y + length * math.sin(angle))
        strands.append((ex, ey))

    return {
        'x': x,
        'y': y,
        'radius': radius,
        'strands': strands,
        'time': time.time()
    }

def draw_stuck_web(img, web, alpha_fade):
    """Renders a stuck web splat on screen with smooth translucent fading."""
    cx, cy = web['x'], web['y']
    overlay = img.copy()

    # Central web splat core
    cv2.circle(overlay, (cx, cy), 18, (255, 255, 255), -1)
    cv2.circle(overlay, (cx, cy), 28, (200, 200, 255), 3)

    # Radial web fracture strands
    for ex, ey in web['strands']:
        cv2.line(overlay, (cx, cy), (ex, ey), (255, 255, 255), 2)
        mid_x = (cx + ex) // 2
        mid_y = (cy + ey) // 2
        cv2.line(overlay, (mid_x - 10, mid_y), (mid_x + 10, mid_y), (220, 220, 255), 1)

    cv2.addWeighted(overlay, alpha_fade, img, 1 - alpha_fade, 0, img)

# Camera setup
cap = cv2.VideoCapture(0)
last_shot_time = 0

print("Spider-Man Web Shooter with Hand Landmarks Active! Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # DRAW HAND LANDMARK DOTS & SKELETON
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=5),    # Crimson Red Joint Dots
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)                 # Pure White Skeleton Lines
            )

            # Key positions
            palm_x = int(hand_landmarks.landmark[9].x * w)
            palm_y = int(hand_landmarks.landmark[9].y * h)
            wrist_x = int(hand_landmarks.landmark[0].x * w)
            wrist_y = int(hand_landmarks.landmark[0].y * h)

            # 1. OPEN PALM GESTURE -> Full Spider Web appears on Palm
            if is_open_palm(hand_landmarks):
                palm_size = int(math.hypot(palm_x - wrist_x, palm_y - wrist_y) * 1.5)
                draw_hand_palm_web(frame, (palm_x, palm_y), max(40, palm_size))
                
                cv2.putText(frame, "WEB CHARGED", (palm_x - 65, palm_y - palm_size - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

            # 2. SPIDER-MAN GESTURE -> Thwip & Splat onto Screen
            elif is_spiderman_gesture(hand_landmarks):
                # Cooldown to prevent spamming splats (0.4s)
                if current_time - last_shot_time > 0.4:
                    dx = palm_x - wrist_x
                    dy = palm_y - wrist_y
                    norm = math.hypot(dx, dy) + 1e-5
                    
                    target_x = int(w / 2 + (dx / norm) * 180)
                    target_y = int(h / 2 + (dy / norm) * 180)

                    stuck_screen_webs.append(create_screen_splat(target_x, target_y))
                    last_shot_time = current_time

                # Draw main web line shooting into screen target
                if stuck_screen_webs:
                    latest = stuck_screen_webs[-1]
                    cv2.line(frame, (wrist_x, wrist_y), (latest['x'], latest['y']), (255, 255, 255), 5)
                    cv2.line(frame, (wrist_x, wrist_y), (latest['x'], latest['y']), (200, 200, 255), 2)

                cv2.putText(frame, "* THWIP! *", (wrist_x - 50, wrist_y - 25),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

    # 3. RENDER STUCK WEBS ON SCREEN (Fades over 4s)
    active_webs = []
    for web in stuck_screen_webs:
        age = current_time - web['time']
        if age < 4.0:
            alpha = max(0.0, 1.0 - (age / 4.0))
            draw_stuck_web(frame, web, alpha)
            active_webs.append(web)

    stuck_screen_webs = active_webs

    cv2.imshow('Spider-Man Tech Web Shooter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()