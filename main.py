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

stuck_screen_webs = []

def is_open_palm(hand_landmarks):
    """Detects fully open palm."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y for tip, pip in zip(tips, pips))

def is_spiderman_gesture(hand_landmarks):
    """Detects Spider-Man gesture."""
    index_open = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    pinky_open = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    middle_closed = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_closed = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    return index_open and pinky_open and middle_closed and ring_closed

def draw_hand_palm_web(img, center, radius):
    """Draws a complete realistic cobweb matrix over the open palm."""
    cx, cy = center
    num_spokes = 8
    rings = 4

    cv2.circle(img, (cx, cy), int(radius * 1.1), (255, 255, 255), 1)

    spoke_angles = [i * (2 * math.pi / num_spokes) for i in range(num_spokes)]
    
    # Draw spokes
    for angle in spoke_angles:
        sx = int(cx + radius * math.cos(angle))
        sy = int(cy + radius * math.sin(angle))
        cv2.line(img, (cx, cy), (sx, sy), (240, 240, 255), 2)

    # Curved spider web rings
    for r in range(1, rings + 1):
        r_dist = (radius / rings) * r
        pts = []
        for angle in spoke_angles:
            rx = int(cx + r_dist * math.cos(angle))
            ry = int(cy + r_dist * math.sin(angle))
            pts.append((rx, ry))

        for i in range(num_spokes):
            p1 = pts[i]
            p2 = pts[(i + 1) % num_spokes]
            mid_x = int((p1[0] + p2[0]) / 2 * 0.9 + cx * 0.1)
            mid_y = int((p1[1] + p2[1]) / 2 * 0.9 + cy * 0.1)
            cv2.polylines(img, [np.array([p1, (mid_x, mid_y), p2])], False, (220, 220, 255), 1, cv2.LINE_AA)

def draw_cinematic_web_stream(img, start_pt, end_pt):
    """
    Renders realistic fluid web throwing with sine-wave turbulence 
    and air particles along the trajectory.
    """
    sx, sy = start_pt
    ex, ey = end_pt

    dist = math.hypot(ex - sx, ey - sy)
    if dist < 10:
        return

    num_pts = 20
    angle = math.atan2(ey - sy, ex - sx)
    perp_angle = angle + math.pi / 2

    # Generate turbulent wavy web path points
    pts = []
    for i in range(num_pts + 1):
        t = i / num_pts
        # Base straight position
        px = sx + (ex - sx) * t
        py = sy + (ey - sy) * t

        # Wave displacement (higher amplitude in middle of the stream)
        wave = math.sin(t * math.pi * 3 + time.time() * 25) * (12 * math.sin(t * math.pi))
        
        final_x = int(px + wave * math.cos(perp_angle))
        final_y = int(py + wave * math.sin(perp_angle))
        pts.append((final_x, final_y))

    # Convert to NumPy array for smooth polyline rendering
    pts_array = np.array(pts, np.int32)

    # 1. Outer Glow / Secondary Strands
    cv2.polylines(img, [pts_array], False, (200, 220, 255), 4, cv2.LINE_AA)
    # 2. Bright White Core Web Filament
    cv2.polylines(img, [pts_array], False, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Flying Web Fluid Droplets / Air Particles along stream
    for i in range(1, num_pts, 2):
        pt = pts[i]
        if random.random() > 0.4:
            offset_x = pt[0] + random.randint(-12, 12)
            offset_y = pt[1] + random.randint(-12, 12)
            cv2.circle(img, (offset_x, offset_y), random.randint(1, 3), (240, 240, 255), -1)
            cv2.line(img, pt, (offset_x, offset_y), (220, 220, 255), 1, cv2.LINE_AA)

def create_photorealistic_splat(x, y):
    """Generates procedural parameters for realistic web glass impact."""
    num_spokes = random.randint(9, 13)
    max_radius = random.randint(150, 220)
    
    spokes = []
    for i in range(num_spokes):
        angle = i * (2 * math.pi / num_spokes) + random.uniform(-0.15, 0.15)
        length = random.uniform(max_radius * 0.6, max_radius)
        ex = int(x + length * math.cos(angle))
        ey = int(y + length * math.sin(angle))
        spokes.append((ex, ey, angle, length))

    # Splat micro-droplets
    droplets = []
    for _ in range(random.randint(18, 28)):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(20, max_radius * 1.1)
        dx = int(x + dist * math.cos(angle))
        dy = int(y + dist * math.sin(angle))
        size = random.randint(2, 5)
        droplets.append((dx, dy, size))

    return {
        'x': x,
        'y': y,
        'radius': max_radius,
        'spokes': spokes,
        'droplets': droplets,
        'time': time.time()
    }

def draw_photorealistic_stuck_web(img, web, alpha):
    """Renders layered 3D depth, drop shadows, and web curves."""
    cx, cy = web['x'], web['y']
    spokes = web['spokes']
    num_spokes = len(spokes)

    overlay = img.copy()

    # 1. Dark Drop Shadow (3D depth effect)
    shadow_offset = 3
    cv2.circle(overlay, (cx + shadow_offset, cy + shadow_offset), 24, (20, 20, 20), -1)
    for ex, ey, _, _ in spokes:
        cv2.line(overlay, (cx + shadow_offset, cy + shadow_offset), 
                 (ex + shadow_offset, ey + shadow_offset), (30, 30, 30), 3, cv2.LINE_AA)

    # 2. Inner Concave Web Rings
    rings = 4
    for r in range(1, rings + 1):
        ring_pts = []
        for ex, ey, angle, length in spokes:
            r_len = (length / rings) * r
            rx = int(cx + r_len * math.cos(angle))
            ry = int(cy + r_len * math.sin(angle))
            ring_pts.append((rx, ry))

        for i in range(num_spokes):
            p1 = ring_pts[i]
            p2 = ring_pts[(i + 1) % num_spokes]
            
            ctrl_x = int((p1[0] + p2[0]) * 0.45 + cx * 0.1)
            ctrl_y = int((p1[1] + p2[1]) * 0.45 + cy * 0.1)
            
            curve_pts = np.array([p1, (ctrl_x, ctrl_y), p2], np.int32)
            cv2.polylines(overlay, [curve_pts], False, (240, 240, 255), 2, cv2.LINE_AA)

    # 3. Main Radial Spokes & Core Splat
    for ex, ey, _, _ in spokes:
        cv2.line(overlay, (cx, cy), (ex, ey), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(overlay, (cx, cy), (ex, ey), (200, 220, 255), 1, cv2.LINE_AA)

    cv2.circle(overlay, (cx, cy), 22, (255, 255, 255), -1)
    cv2.circle(overlay, (cx, cy), 30, (210, 210, 255), 3, cv2.LINE_AA)

    # 4. Splat Droplets
    for dx, dy, size in web['droplets']:
        cv2.circle(overlay, (dx, dy), size, (255, 255, 255), -1)

    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

# Camera setup
cap = cv2.VideoCapture(0)
last_shot_time = 0

print("Cinematic Spider-Man Web Shooter Active. Press 'q' to quit.")

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
            # Draw HUD Hand skeleton dots and lines
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            palm_x = int(hand_landmarks.landmark[9].x * w)
            palm_y = int(hand_landmarks.landmark[9].y * h)
            wrist_x = int(hand_landmarks.landmark[0].x * w)
            wrist_y = int(hand_landmarks.landmark[0].y * h)

            # 1. Open Palm -> Web Matrix Charges
            if is_open_palm(hand_landmarks):
                palm_size = int(math.hypot(palm_x - wrist_x, palm_y - wrist_y) * 1.5)
                draw_hand_palm_web(frame, (palm_x, palm_y), max(40, palm_size))

            # 2. Spider-Man Gesture -> Turbulent Web Shoot & Screen Impact
            elif is_spiderman_gesture(hand_landmarks):
                if current_time - last_shot_time > 0.4:
                    dx = palm_x - wrist_x
                    dy = palm_y - wrist_y
                    norm = math.hypot(dx, dy) + 1e-5
                    
                    target_x = int(w / 2 + (dx / norm) * 180)
                    target_y = int(h / 2 + (dy / norm) * 180)

                    stuck_screen_webs.append(create_photorealistic_splat(target_x, target_y))
                    last_shot_time = current_time

                if stuck_screen_webs:
                    latest = stuck_screen_webs[-1]
                    draw_cinematic_web_stream(frame, (wrist_x, wrist_y), (latest['x'], latest['y']))

    # Render Stuck Screen Webs
    active_webs = []
    for web in stuck_screen_webs:
        age = current_time - web['time']
        if age < 4.5:
            alpha = max(0.0, 1.0 - (age / 4.5))
            draw_photorealistic_stuck_web(frame, web, alpha)
            active_webs.append(web)

    stuck_screen_webs = active_webs

    cv2.imshow('Spider-Man Web Shooter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()