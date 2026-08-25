import cv2
import mediapipe as mp
import numpy as np
import math
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("model is done.")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)



def get_palm_center(hand_landmarks, w, h):
    ids = [0, 5, 9, 13, 17]
    xs = [hand_landmarks[i].x for i in ids]
    ys = [hand_landmarks[i].y for i in ids]
    return int(np.mean(xs) * w), int(np.mean(ys) * h)


def get_index_tip(hand_landmarks, w, h):
    lm = hand_landmarks[8]  
    return int(lm.x * w), int(lm.y * h)



def generate_random_polygon(num_points=8, seed_offset=0):
    rng = random.Random(seed_offset)
    angles = sorted([rng.uniform(0, 2 * math.pi) for _ in range(num_points)])
    radii_ratio = [rng.uniform(0.6, 1.0) for _ in range(num_points)]
    return list(zip(angles, radii_ratio))


def normalize_drawn_shape(points):
    pts = np.array(points, dtype=np.float32)
    center = pts.mean(axis=0)
    shifted = pts - center
    max_r = np.max(np.linalg.norm(shifted, axis=1))
    if max_r == 0:
        max_r = 1
    result = []
    for x, y in shifted:
        theta = math.atan2(y, x)
        r_ratio = np.linalg.norm([x, y]) / max_r
        result.append((theta, r_ratio))
    result.sort(key=lambda p: p[0])
    return result


def build_polygon_points(center, radius, angle_offset, shape_def):
    pts = []
    for theta, r_ratio in shape_def:
        a = theta + angle_offset
        x = int(center[0] + radius * r_ratio * math.cos(a))
        y = int(center[1] + radius * r_ratio * math.sin(a))
        pts.append([x, y])
    return np.array(pts, np.int32)



def apply_holographic(frame, t):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    y_grad = np.linspace(0, 1, h).reshape(h, 1)
    y_grad = np.repeat(y_grad, w, axis=1)

    hue = (gray * 0.5 + y_grad * 0.5 + t) % 1.0
    sat = np.full_like(hue, 0.85)
    val = np.clip(0.5 + gray * 0.5, 0, 1)

    hsv = np.stack([hue * 179, sat * 255, val * 255], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr



def draw_checkerboard_colorful(canvas, mask, t, cell_size=25, alpha=0.45):
    h, w = mask.shape
    overlay = np.zeros_like(canvas)
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            idx = (x // cell_size) + (y // cell_size)
            hue = int((idx * 25 + t * 60) % 180)
            color_hsv = np.uint8([[[hue, 220, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            overlay[y:y + cell_size, x:x + cell_size] = color_bgr.tolist()
    blended = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    canvas[mask == 255] = blended[mask == 255]


def draw_shape_icon(img, shape_type, center, size, color):
    if shape_type == 0:  
        pts = []
        for i in range(10):
            angle = i * math.pi / 5 - math.pi / 2
            r = size if i % 2 == 0 else size * 0.45
            x = int(center[0] + r * math.cos(angle))
            y = int(center[1] + r * math.sin(angle))
            pts.append([x, y])
        cv2.fillPoly(img, [np.array(pts, np.int32)], color)
    elif shape_type == 1:  
        cv2.circle(img, center, int(size * 0.6), color, -1)
    elif shape_type == 2:  
        x0, y0 = center[0] - size // 2, center[1] - int(size * 0.35)
        x1, y1 = center[0] + size // 2, center[1] + int(size * 0.35)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
    elif shape_type == 3:  
        pts = np.array([
            [center[0], center[1] - size // 2],
            [center[0] - size // 2, center[1] + size // 2],
            [center[0] + size // 2, center[1] + size // 2]
        ], np.int32)
        cv2.fillPoly(img, [pts], color)


def draw_random_shapes_pattern(canvas, mask, bbox, shape_cycle_index, alpha=0.5, seed=7):
    x0, y0, x1, y1 = bbox
    overlay = canvas.copy()
    rng = random.Random(seed)
    step = 45
    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            local_shape = (shape_cycle_index + rng.randint(0, 3)) % 4
            size = rng.randint(14, 22)
            draw_shape_icon(overlay, local_shape, (x, y), size, (255, 255, 255))
    blended = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    canvas[mask == 255] = blended[mask == 255]


def draw_shape_on_frame(frame, pts, effect_frame, pattern_mode, t, shape_cycle_index, alpha=0.85):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    blended = cv2.addWeighted(effect_frame, alpha, frame, 1 - alpha, 0)
    frame[mask == 255] = blended[mask == 255]

    x, y, w, h = cv2.boundingRect(pts)
    bbox = (x, y, x + w, y + h)

    if pattern_mode == 1:
        draw_checkerboard_colorful(frame, mask, t)
    elif pattern_mode == 2:
        draw_random_shapes_pattern(frame, mask, bbox, shape_cycle_index)

    cv2.polylines(frame, [pts], True, (255, 255, 255), 2)




cap = cv2.VideoCapture(0)

current_shape = generate_random_polygon(num_points=8, seed_offset=42)
drawing_mode = False
drawn_points = []
t = 0.0
frame_count = 0

pattern_mode = 0  
shape_cycle_index = 0
shape_change_interval = 40  

print("کلیدها:")
print("  d = شروع رسم شکل با انگشت اشاره")
print("  f = پایان رسم (شکل کشیده‌شده جایگزین میشه)")
print("  r = شکل رندوم جدید")
print("  p = تغییر حالت (هولوگرام / شطرنجی رنگی / اشکال رندوم متغیر)")
print("  q = خروج")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    hands = result.hand_landmarks if result.hand_landmarks else []

    if drawing_mode:
        cv2.putText(frame, "Drawing... move index finger, press 'f' to finish",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if len(hands) >= 1:
            tip = get_index_tip(hands[0], w, h)
            drawn_points.append(tip)
        if len(drawn_points) > 1:
            pts_draw = np.array(drawn_points, np.int32)
            cv2.polylines(frame, [pts_draw], False, (0, 255, 255), 2)

    elif len(hands) == 2:
        p1 = get_palm_center(hands[0], w, h)
        p2 = get_palm_center(hands[1], w, h)
        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        radius = int(dist / 1.8)
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        if frame_count % shape_change_interval == 0:
            shape_cycle_index = (shape_cycle_index + 1) % 4

        pts = build_polygon_points(center, radius, angle, current_shape)
        effect_frame = apply_holographic(frame, t)
        draw_shape_on_frame(frame, pts, effect_frame, pattern_mode, t, shape_cycle_index)

        cv2.circle(frame, p1, 6, (0, 255, 0), -1)
        cv2.circle(frame, p2, 6, (0, 255, 0), -1)
    else:
        cv2.putText(frame, "Show both hands", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    t += 0.004
    frame_count += 1
    cv2.imshow("Hand Shape Hologram", frame)      
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        drawing_mode = True
        drawn_points = []
    elif key == ord('f'):
        if len(drawn_points) > 5:
            current_shape = normalize_drawn_shape(drawn_points)
        drawing_mode = False
    elif key == ord('r'):
        current_shape = generate_random_polygon(num_points=8, seed_offset=random.randint(0, 9999))
    elif key == ord('p'):
        pattern_mode = (pattern_mode + 1) % 3

cap.release()
cv2.destroyAllWindows()


