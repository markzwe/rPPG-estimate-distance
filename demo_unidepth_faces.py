"""
demo_unidepth_faces.py

Lightweight InsightFace + UniDepth demo that estimates per-face landmark depth.
Edit FOLDER_PATH below, then run:

    python demo_unidepth_faces.py
"""

import cv2
import os
import torch
import numpy as np
import time
import csv
from insightface.app import FaceAnalysis
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole

# --- Helper: focal length ---
def getFocalLength_px(focal_length_mm, img_w, sensor_w=36):
    return (focal_length_mm * img_w) / sensor_w

# --- Helper: safe patch extraction ---
def get_patch(depth_map, x, y, w, h, size=2):
    y1 = max(y - size, 0)
    y2 = min(y + size + 1, h)
    x1 = max(x - size, 0)
    x2 = min(x + size + 1, w)
    return depth_map[y1:y2, x1:x2]

# --- SETTINGS ---
FOLDER_PATH = "assets/01-02"
OUTPUT_CSV = "distance_results5m.csv"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ORIGINAL_W = 1280
TARGET_W, TARGET_H = 854, 480
SCALE = TARGET_W / ORIGINAL_W
FOCAL_LENGTH_PX = getFocalLength_px(26, 4032) * SCALE

# --- INIT MODELS ---
face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(DEVICE)
model.eval()

# --- LOAD IMAGES ---
image_files = sorted([
    f for f in os.listdir(FOLDER_PATH)
    if f.lower().endswith(('.png', '.jpg'))
])

results_data = []
prev_time = 0
num = 0

# --- MAIN LOOP ---
for filename in image_files:
    img_path = os.path.join(FOLDER_PATH, filename)
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        continue

    num += 1
    print(f"Processing image {num} of {len(image_files)}: {filename}")

    img_bgr = cv2.resize(raw_img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Prepare input ---
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float() / 255.0

    intrinsics = torch.tensor([
        [
            [FOCAL_LENGTH_PX, 0, w / 2],
            [0, FOCAL_LENGTH_PX, h / 2],
            [0, 0, 1]
        ]
    ]).float()

    camera = Pinhole(K=intrinsics).to(DEVICE)

    # --- Inference ---
    faces = face_app.get(img_bgr)

    with torch.no_grad():
        predictions = model.infer(input_tensor, camera=camera)
        depth_map = predictions["depth"][0, 0].cpu().numpy()

    # --- Process each face ---
    for i, face in enumerate(faces):
        bx1, by1, bx2, by2 = face.bbox.astype(int)

        # --- Landmarks ---
        landmarks = face.kps
        L_eye, R_eye, nose, L_mouth, R_mouth = landmarks

        L_eye = tuple(map(int, L_eye))
        R_eye = tuple(map(int, R_eye))
        nose = tuple(map(int, nose))
        L_mouth = tuple(map(int, L_mouth))
        R_mouth = tuple(map(int, R_mouth))

        # --- Simple landmark-point ROIs ---
        point_size = 2

        nose_roi = get_patch(depth_map, nose[0], nose[1], w, h, size=point_size)
        left_eye_roi = get_patch(depth_map, L_eye[0], L_eye[1], w, h, size=point_size)
        right_eye_roi = get_patch(depth_map, R_eye[0], R_eye[1], w, h, size=point_size)
        left_mouth_roi = get_patch(depth_map, L_mouth[0], L_mouth[1], w, h, size=point_size)
        right_mouth_roi = get_patch(depth_map, R_mouth[0], R_mouth[1], w, h, size=point_size)

        roi_depths = {}
        if nose_roi.size > 0:
            roi_depths["nose"] = float(np.median(nose_roi))
        if left_eye_roi.size > 0:
            roi_depths["left_eye"] = float(np.median(left_eye_roi))
        if right_eye_roi.size > 0:
            roi_depths["right_eye"] = float(np.median(right_eye_roi))
        if left_mouth_roi.size > 0:
            roi_depths["left_mouth"] = float(np.median(left_mouth_roi))
        if right_mouth_roi.size > 0:
            roi_depths["right_mouth"] = float(np.median(right_mouth_roi))

        if not roi_depths:
            continue

        dist_m = float(np.median(list(roi_depths.values())))
        nose_d = roi_depths.get("nose", np.nan)
        left_eye_d = roi_depths.get("left_eye", np.nan)
        right_eye_d = roi_depths.get("right_eye", np.nan)
        left_mouth_d = roi_depths.get("left_mouth", np.nan)
        right_mouth_d = roi_depths.get("right_mouth", np.nan)

        print(f"Face {i+1} distance: {dist_m:.2f}m")
        print(f"""
        Nose: {nose_d:.2f}m
        Left Eye: {left_eye_d:.2f}m
        Right Eye: {right_eye_d:.2f}m
        Left Mouth: {left_mouth_d:.2f}m
        Right Mouth: {right_mouth_d:.2f}m
        Final: {dist_m:.2f}m
        """)
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        # --- Draw landmarks ---
        for (x, y) in landmarks:
            x, y = int(x), int(y)
            cv2.circle(img_bgr, (x, y), 3, (0, 0, 255), -1)

        # --- Draw ROI boxes (debug) ---
        cv2.rectangle(img_bgr, (nose[0]-2, nose[1]-2), (nose[0]+2, nose[1]+2), (255, 0, 0), 1)
        cv2.rectangle(img_bgr, (L_eye[0]-2, L_eye[1]-2), (L_eye[0]+2, L_eye[1]+2), (0, 255, 255), 1)
        cv2.rectangle(img_bgr, (R_eye[0]-2, R_eye[1]-2), (R_eye[0]+2, R_eye[1]+2), (0, 255, 255), 1)
        cv2.rectangle(img_bgr, (L_mouth[0]-2, L_mouth[1]-2), (L_mouth[0]+2, L_mouth[1]+2), (255, 0, 255), 1)
        cv2.rectangle(img_bgr, (R_mouth[0]-2, R_mouth[1]-2), (R_mouth[0]+2, R_mouth[1]+2), (255, 0, 255), 1)

        # --- Draw bounding box ---
        cv2.rectangle(img_bgr, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{dist_m:.2f}m",
                    (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

        # --- Save CSV ---
        results_data.append([
            filename,
            round(float(dist_m), 3),
            bx1,
            by1,
            bx2,
            by2,
            round(float(nose_d), 3) if not np.isnan(nose_d) else "",
            round(float(left_eye_d), 3) if not np.isnan(left_eye_d) else "",
            round(float(right_eye_d), 3) if not np.isnan(right_eye_d) else "",
            round(float(left_mouth_d), 3) if not np.isnan(left_mouth_d) else "",
            round(float(right_mouth_d), 3) if not np.isnan(right_mouth_d) else "",
        ])

    # --- FPS ---
    end_time = time.time()
    fps = 1 / (end_time - prev_time) if (end_time - prev_time) > 0 else 0
    prev_time = end_time

    cv2.putText(img_bgr, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # --- Side-by-side visualization ---
    if depth_color.shape[:2] != img_bgr.shape[:2]:
        depth_color = cv2.resize(depth_color, (img_bgr.shape[1], img_bgr.shape[0]))

    combined = np.hstack((img_bgr, depth_color))
    cv2.imshow("RGB | Depth", combined)

    if cv2.waitKey(1) & 0xFF == ord('q') or num == 10:
        break

# --- SAVE CSV ---
header = ['filename', 'distance_m', 'x1', 'y1', 'x2', 'y2', 'nose_d', 'left_eye_d', 'right_eye_d', 'left_mouth_d', 'right_mouth_d']

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(results_data)

print(f"Data successfully saved to {OUTPUT_CSV}")
cv2.destroyAllWindows()