import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random

# Access MediaPipe solutions
try:
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# Setup Models
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

stuck_screen_webs = []

# Suit Color Themes (BGR)
SUIT_THEMES = {
    '1': {
        'name': 'Classic Red',
        'mask_color': (25, 25, 195),
        'rim_color': (15, 15, 15),
        'lens_color': (245, 245, 245),
        'web_color': (15, 15, 15),
        'alpha': 0.65
    },
    '2': {
        'name': 'Miles Morales (Stealth)',
        'mask_color': (18, 18, 22),
        'rim_color': (15, 15, 210),
        'lens_color': (240, 240, 250),
        'web_color': (20, 20, 180),
        'alpha': 0.85
    },
    '3': {
        'name': 'Iron Spider (Armor)',
        'mask_color': (20, 20, 150),
        'rim_color': (30, 190, 230),
        'lens_color': (220, 250, 255),
        'web_color': (20, 160, 200),
        'alpha': 0.70
    },
    '4': {
        'name': 'Symbiote Black',
        'mask_color': (10, 10, 12),
        'rim_color': (60, 60, 65),
        'lens_color': (220, 220, 220),
        'web_color': (40, 40, 45),
        'alpha': 0.90
    }
}

current_suit_key = '1'

def is_open_palm(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y for tip, pip in zip(tips, pips))

def is_spiderman_gesture(hand_landmarks):
    index_open = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    pinky_open = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    middle_closed = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_closed = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    return index_open and pinky_open and middle_closed and ring_closed

def build_angular_spidey_lens(inner_corner, top_brow, outer_temple, bottom_cheek):
    """
    Constructs a stylized sharp, upward-swept 6-point Spider-Man lens polygon 
    from facial anchor points.
    """
    ix, iy = inner_corner
    tx, ty = top_brow
    ox, oy = outer_temple
    bx, by = bottom_cheek

    # Sharp top sweep toward temple
    p1 = (ix, iy)
    p2 = (int(ix * 0.4 + tx * 0.6), int(ty - 10))
    p3 = (int(ox + 15), int(oy - 20)) # Sharp outer top wing tip
    p4 = (int(ox + 5), int(oy + 15))   # Outer lower edge
    p5 = (int(bx), int(by + 10))       # Bottom curve point
    p6 = (int(ix * 0.7 + bx * 0.3), int(iy * 0.5 + by * 0.5))

    return np.array([p1, p2, p3, p4, p5, p6], np.int32)

def draw_photorealistic_mask(img, face_landmarks, h, w, theme):
    pts = face_landmarks.landmark

    # 1. Base Mask Red Overlay
    jaw_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    face_contour = np.array([(int(pts[i].x * w), int(pts[i].y * h)) for i in jaw_indices], np.int32)

    mask_layer = img.copy()
    cv2.fillPoly(mask_layer, [face_contour], theme['mask_color'])
    cv2.addWeighted(mask_layer, theme['alpha'], img, 1.0 - theme['alpha'], 0, img)

    # 2. Extract Structural Eye Anchors
    # Left Eye Anchors
    l_inner = (int(pts[133].x * w), int(pts[133].y * h))
    l_top = (int(pts[159].x * w), int(pts[159].y * h))
    l_outer = (int(pts[33].x * w), int(pts[33].y * h))
    l_bottom = (int(pts[145].x * w), int(pts[145].y * h))

    # Right Eye Anchors
    r_inner = (int(pts[362].x * w), int(pts[362].y * h))
    r_top = (int(pts[386].x * w), int(pts[386].y * h))
    r_outer = (int(pts[263].x * w), int(pts[263].y * h))
    r_bottom = (int(pts[374].x * w), int(pts[374].y * h))

    # Generate Left & Right Angular Spider-Man Lenses
    l_lens = build_angular_spidey_lens(l_inner, l_top, l_outer, l_bottom)
    r_lens = build_angular_spidey_lens(r_inner, r_top, r_outer, r_bottom)

    # Function to scale polygon around its centroid for rim thickness
    def scale_lens(poly, factor):
        centroid = np.mean(poly, axis=0)
        return np.int32((poly - centroid) * factor + centroid)

    l_lens_outer = scale_lens(l_lens, 1.35)
    r_lens_outer = scale_lens(r_lens, 1.35)

    l_lens_mid = scale_lens(l_lens, 1.15)
    r_lens_mid = scale_lens(r_lens, 1.15)

    # Draw Outer Black/Gold Heavy Bevel Frame
    cv2.fillPoly(img, [l_lens_outer], theme['rim_color'])
    cv2.fillPoly(img, [r_lens_outer], theme['rim_color'])

    # Inner Lens Base White/Silver Fill
    cv2.fillPoly(img, [l_lens], theme['lens_color'])
    cv2.fillPoly(img, [r_lens], theme['lens_color'])

    # Sharp Rim Lines & Glass Reflections
    cv2.polylines(img, [l_lens_outer], True, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.polylines(img, [r_lens_outer], True, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.polylines(img, [l_lens], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(img, [r_lens], True, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Facial Cobweb Mesh Lines
    nose_x, nose_y = int(pts[1].x * w), int(pts[1].y * h)
    web_anchors = [10, 109, 338, 297, 356, 454, 361, 152, 132, 234, 127, 67]
    
    anchor_coords = []
    for idx in web_anchors:
        ax, ay = int(pts[idx].x * w), int(pts[idx].y * h)
        anchor_coords.append((ax, ay))
        cv2.line(img, (nose_x, nose_y), (ax, ay), theme['web_color'], 1, cv2.LINE_AA)

    num_anchors = len(anchor_coords)
    for r in [0.30, 0.55, 0.80]:
        ring_pts = [(int(nose_x + (ax - nose_x) * r), int(nose_y + (ay - nose_y) * r)) for ax, ay in anchor_coords]

        for i in range(num_anchors):
            p1, p2 = ring_pts[i], ring_pts[(i + 1) % num_anchors]
            ctrl_x = int((p1[0] + p2[0]) * 0.45 + nose_x * 0.1)
            ctrl_y = int((p1[1] + p2[1]) * 0.45 + nose_y * 0.1)
            cv2.polylines(img, [np.array([p1, (ctrl_x, ctrl_y), p2], np.int32)], False, theme['web_color'], 1, cv2.LINE_AA)

def draw_hand_palm_web(img, center, radius):
    cx, cy = center
    num_spokes, rings = 8, 4

    cv2.circle(img, (cx, cy), int(radius * 1.1), (255, 255, 255), 1)
    spoke_angles = [i * (2 * math.pi / num_spokes) for i in range(num_spokes)]

    for angle in spoke_angles:
        sx = int(cx + radius * math.cos(angle))
        sy = int(cy + radius * math.sin(angle))
        cv2.line(img, (cx, cy), (sx, sy), (240, 240, 255), 2)

    for r in range(1, rings + 1):
        r_dist = (radius / rings) * r
        pts = [(int(cx + r_dist * math.cos(a)), int(cy + r_dist * math.sin(a))) for a in spoke_angles]

        for i in range(num_spokes):
            p1, p2 = pts[i], pts[(i + 1) % num_spokes]
            mid_x = int((p1[0] + p2[0]) / 2 * 0.9 + cx * 0.1)
            mid_y = int((p1[1] + p2[1]) / 2 * 0.9 + cy * 0.1)
            cv2.polylines(img, [np.array([p1, (mid_x, mid_y), p2])], False, (220, 220, 255), 1, cv2.LINE_AA)

def draw_cinematic_web_stream(img, start_pt, end_pt):
    sx, sy = start_pt
    ex, ey = end_pt
    dist = math.hypot(ex - sx, ey - sy)
    if dist < 10:
        return

    num_pts = 20
    angle = math.atan2(ey - sy, ex - sx)
    perp_angle = angle + math.pi / 2

    pts = []
    for i in range(num_pts + 1):
        t = i / num_pts
        px, py = sx + (ex - sx) * t, sy + (ey - sy) * t
        wave = math.sin(t * math.pi * 3 + time.time() * 25) * (12 * math.sin(t * math.pi))
        pts.append((int(px + wave * math.cos(perp_angle)), int(py + wave * math.sin(perp_angle))))

    pts_array = np.array(pts, np.int32)
    cv2.polylines(img, [pts_array], False, (200, 220, 255), 4, cv2.LINE_AA)
    cv2.polylines(img, [pts_array], False, (255, 255, 255), 2, cv2.LINE_AA)

    for i in range(1, num_pts, 2):
        pt = pts[i]
        if random.random() > 0.4:
            offset_x, offset_y = pt[0] + random.randint(-12, 12), pt[1] + random.randint(-12, 12)
            cv2.circle(img, (offset_x, offset_y), random.randint(1, 3), (240, 240, 255), -1)
            cv2.line(img, pt, (offset_x, offset_y), (220, 220, 255), 1, cv2.LINE_AA)

def create_photorealistic_splat(x, y):
    num_spokes = random.randint(9, 13)
    max_radius = random.randint(150, 220)
    spokes = []
    for i in range(num_spokes):
        angle = i * (2 * math.pi / num_spokes) + random.uniform(-0.15, 0.15)
        length = random.uniform(max_radius * 0.6, max_radius)
        spokes.append((int(x + length * math.cos(angle)), int(y + length * math.sin(angle)), angle, length))

    droplets = [(int(x + random.uniform(20, max_radius * 1.1) * math.cos(a := random.uniform(0, 2 * math.pi))),
                 int(y + random.uniform(20, max_radius * 1.1) * math.sin(a)),
                 random.randint(2, 5)) for _ in range(random.randint(18, 28))]

    return {'x': x, 'y': y, 'radius': max_radius, 'spokes': spokes, 'droplets': droplets, 'time': time.time()}

def draw_photorealistic_stuck_web(img, web, alpha):
    cx, cy, spokes = web['x'], web['y'], web['spokes']
    num_spokes = len(spokes)
    overlay = img.copy()

    shadow_offset = 3
    cv2.circle(overlay, (cx + shadow_offset, cy + shadow_offset), 24, (20, 20, 20), -1)
    for ex, ey, _, _ in spokes:
        cv2.line(overlay, (cx + shadow_offset, cy + shadow_offset), (ex + shadow_offset, ey + shadow_offset), (30, 30, 30), 3, cv2.LINE_AA)

    rings = 4
    for r in range(1, rings + 1):
        ring_pts = [(int(cx + (length / rings) * r * math.cos(angle)), int(cy + (length / rings) * r * math.sin(angle))) for _, _, angle, length in spokes]
        for i in range(num_spokes):
            p1, p2 = ring_pts[i], ring_pts[(i + 1) % num_spokes]
            ctrl_x, ctrl_y = int((p1[0] + p2[0]) * 0.45 + cx * 0.1), int((p1[1] + p2[1]) * 0.45 + cy * 0.1)
            cv2.polylines(overlay, [np.array([p1, (ctrl_x, ctrl_y), p2], np.int32)], False, (240, 240, 255), 2, cv2.LINE_AA)

    for ex, ey, _, _ in spokes:
        cv2.line(overlay, (cx, cy), (ex, ey), (255, 255, 255), 3, cv2.LINE_AA)

    cv2.circle(overlay, (cx, cy), 22, (255, 255, 255), -1)
    for dx, dy, size in web['droplets']:
        cv2.circle(overlay, (dx, dy), size, (255, 255, 255), -1)

    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

# Main Application Loop
cap = cv2.VideoCapture(0)
last_shot_time = 0

print("Spider-Man Mask Active!")
print("Press 1: Classic Red | 2: Miles Morales | 3: Iron Spider | 4: Symbiote | Q: Quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    active_theme = SUIT_THEMES[current_suit_key]

    # 1. Face Mesh for Angular Mask Lenses
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            draw_photorealistic_mask(frame, face_landmarks, h, w, active_theme)

    # 2. Hand Tracking for Web Shooting
    hand_results = hands.process(rgb_frame)
    current_time = time.time()

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            palm_x, palm_y = int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h)
            wrist_x, wrist_y = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)

            if is_open_palm(hand_landmarks):
                palm_size = int(math.hypot(palm_x - wrist_x, palm_y - wrist_y) * 1.5)
                draw_hand_palm_web(frame, (palm_x, palm_y), max(40, palm_size))

            elif is_spiderman_gesture(hand_landmarks):
                if current_time - last_shot_time > 0.4:
                    dx, dy = palm_x - wrist_x, palm_y - wrist_y
                    norm = math.hypot(dx, dy) + 1e-5
                    target_x, target_y = int(w / 2 + (dx / norm) * 180), int(h / 2 + (dy / norm) * 180)

                    stuck_screen_webs.append(create_photorealistic_splat(target_x, target_y))
                    last_shot_time = current_time

                if stuck_screen_webs:
                    latest = stuck_screen_webs[-1]
                    draw_cinematic_web_stream(frame, (wrist_x, wrist_y), (latest['x'], latest['y']))

    # 3. Render Stuck Screen Webs
    active_webs = []
    for web in stuck_screen_webs:
        age = current_time - web['time']
        if age < 4.5:
            alpha = max(0.0, 1.0 - (age / 4.5))
            draw_photorealistic_stuck_web(frame, web, alpha)
            active_webs.append(web)

    stuck_screen_webs = active_webs

    # HUD Indicator
    cv2.rectangle(frame, (10, 10), (320, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"SUIT: {active_theme['name']} [1-4]", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Spider-Man AR Camera', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif chr(key) in ['1', '2', '3', '4']:
        current_suit_key = chr(key)

cap.release()
cv2.destroyAllWindows()