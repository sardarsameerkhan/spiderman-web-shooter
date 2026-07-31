import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time

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
    """Draws an animated spider web stream from start_point towards target_point."""
    sx, sy = start_point
    tx, ty = target_point

    curr_tx = int(sx + (tx - sx) * progress)
    curr_ty = int(sy + (ty - sy) * progress)

    # Core web lines
    cv2.line(img, (sx, sy), (curr_tx, curr_ty), (255, 255, 255), 4)
    cv2.line(img, (sx, sy), (curr_tx, curr_ty), (200, 200, 255), 1)

    # Branching web strands
    angle = math.atan2(curr_ty - sy, curr_tx - sx)
    length = math.hypot(curr_tx - sx, curr_ty - sy)

    num_branches = 5
    for i in range(1, num_branches + 1):
        dist = (length / num_branches) * i
        bx = int(sx + dist * math.cos(angle))
        by = int(sy + dist * math.sin(angle))

        flare_len = int(15 * (i / num_branches))
        perp_angle1 = angle + math.pi / 3
        perp_angle2 = angle - math.pi / 3

        fx1 = int(bx + flare_len * math.cos(perp_angle1))
        fy1 = int(by + flare_len * math.sin(perp_angle1))
        fx2 = int(bx + flare_len * math.cos(perp_angle2))
        fy2 = int(by + flare_len * math.sin(perp_angle2))

        cv2.line(img, (bx, by), (fx1, fy1), (240, 240, 255), 1)
        cv2.line(img, (bx, by), (fx2, fy2), (240, 240, 255), 1)

    if progress >= 0.9:
        cv2.circle(img, (curr_tx, curr_ty), 12, (255, 255, 255), 2)
        cv2.circle(img, (curr_tx, curr_ty), 5, (0, 255, 255), -1)

    return (curr_tx, curr_ty)

def line_point_distance(line_start, line_end, point):
    """Calculates perpendicular distance between a point and a line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
    if line_len_sq == 0:
        return math.hypot(px - x1, py - y1)

    t = max(0, min(1, ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / line_len_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return math.hypot(px - proj_x, py - proj_y)

# Game State Variables
score = 0
targets = []  # [{ 'pos': [x, y], 'vel': [vx, vy], 'radius': 25, 'hit_time': 0 }]

def spawn_target(w, h):
    x = random.randint(100, w - 100)
    y = random.randint(100, h // 2)
    vx = random.choice([-3, -2, 2, 3])
    vy = random.choice([-2, -1, 1, 2])
    return {'pos': [x, y], 'vel': [vx, vy], 'radius': 30, 'hit_time': None}

# Initialize camera
cap = cv2.VideoCapture(0)
web_animation_frames = {}

print("Spider-Man Game Started! Aim and shoot floating targets with 🤟. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Spawn targets if fewer than 3 on screen
    while len(targets) < 3:
        targets.append(spawn_target(w, h))

    # Update target movement physics
    for t in targets:
        if t['hit_time'] is None:
            t['pos'][0] += t['vel'][0]
            t['pos'][1] += t['vel'][1]

            # Bounce off screen walls
            if t['pos'][0] - t['radius'] <= 0 or t['pos'][0] + t['radius'] >= w:
                t['vel'][0] *= -1
            if t['pos'][1] - t['radius'] <= 0 or t['pos'][1] + t['radius'] >= h - 150:
                t['vel'][1] *= -1

    results = hands.process(rgb_frame)
    current_hands_shooting = []

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )

            if is_spiderman_gesture(hand_landmarks):
                current_hands_shooting.append(idx)

                wrist_x = int(hand_landmarks.landmark[0].x * w)
                wrist_y = int(hand_landmarks.landmark[0].y * h)
                mid_x = int(hand_landmarks.landmark[9].x * w)
                mid_y = int(hand_landmarks.landmark[9].y * h)

                dx = mid_x - wrist_x
                dy = mid_y - wrist_y
                norm = math.hypot(dx, dy) + 1e-5
                dir_x = dx / norm
                dir_y = dy / norm

                target_x = int(wrist_x + dir_x * 900)
                target_y = int(wrist_y + dir_y * 900)

                anim_frame = web_animation_frames.get(idx, 0) + 1
                web_animation_frames[idx] = anim_frame
                progress = min(1.0, anim_frame / 4.0)

                # Draw web line
                web_end = draw_spider_web(frame, (wrist_x, wrist_y), (target_x, target_y), progress)

                # Check collision with targets
                for t in targets:
                    if t['hit_time'] is None:
                        dist = line_point_distance((wrist_x, wrist_y), web_end, t['pos'])
                        if dist <= t['radius'] + 10:
                            t['hit_time'] = time.time()
                            score += 100

                # THWIP Text
                cv2.putText(frame, "THWIP!", (wrist_x - 40, wrist_y - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 3)
                cv2.putText(frame, "THWIP!", (wrist_x - 40, wrist_y - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)

    # Render Targets & Web Hit Effects
    curr_time = time.time()
    active_targets = []

    for t in targets:
        tx, ty = int(t['pos'][0]), int(t['pos'][1])
        if t['hit_time'] is None:
            # Draw active target (Green Target Orb)
            cv2.circle(frame, (tx, ty), t['radius'], (0, 255, 0), -1)
            cv2.circle(frame, (tx, ty), t['radius'] + 4, (255, 255, 255), 2)
            cv2.circle(frame, (tx, ty), 8, (0, 0, 255), -1)
            active_targets.append(t)
        else:
            # Web splat effect on hit
            elapsed = curr_time - t['hit_time']
            if elapsed < 0.4:  # Splat effect duration
                cv2.circle(frame, (tx, ty), t['radius'] + 15, (255, 255, 255), -1)
                cv2.putText(frame, "+100", (tx - 25, ty - 35),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
                active_targets.append(t)

    targets = active_targets
    web_animation_frames = {k: v for k, v in web_animation_frames.items() if k in current_hands_shooting}

    # HUD Banner (Scoreboard)
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(frame, f"SPIDER-MAN WEB SHOOTER | SCORE: {score}", (20, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow('Spider-Man Web Shooter Game', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()