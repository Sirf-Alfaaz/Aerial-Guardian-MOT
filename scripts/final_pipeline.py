from ultralytics import YOLO
import cv2
import time
import numpy as np
from collections import defaultdict
import os

MODEL_PATH = "models/best.pt"

SEQUENCE_FOLDER = r"sequences\uav0000074_11856_v"


OUTPUT_PATH = "outputs/uav0000075_00000_v_tracked.mp4"

CONFIDENCE = 0.15
IOU        = 0.45
IMGSZ      = 832
TAIL_LENGTH = 30
SHOW_WINDOW = True
DETECTION_INTERVAL = 2

os.makedirs("outputs", exist_ok=True)

print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully")


image_files = sorted([
    f for f in os.listdir(SEQUENCE_FOLDER)
    if f.endswith((".jpg", ".png", ".jpeg"))
])

if len(image_files) == 0:
    raise Exception("No images found in sequence folder")

print(f"[INFO] Total Frames: {len(image_files)}")

sample_frame = cv2.imread(os.path.join(SEQUENCE_FOLDER, image_files[0]))
if sample_frame is None:
    raise Exception("Could not read first image")

height, width = sample_frame.shape[:2]
fps_input = 30
print(f"[INFO] Resolution: {width}x{height}")


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_input, (width, height))

track_history   = defaultdict(list)
track_seen_count = defaultdict(int)
last_results    = None

frame_count  = 0
start_total  = time.time()
prev_time    = time.time()

total_detections_ever = set()   


TAIL_COLORS = [
    (
        0,                                  
        int(255 * (1 - i / TAIL_LENGTH)),   
        255                                 
    )
    for i in range(TAIL_LENGTH)
]

def draw_hud(frame, avg_fps, inst_fps, frame_idx, total_frames,
             active_count, total_unique):

    panel_w = 280
    panel_h = 160
    x, y    = 12, 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (0, 200, 100), 8)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    lh    = 28   
    tx    = x + 10
    ty    = y + 26

    lines = [
        (f"FPS  avg : {avg_fps:6.2f}",   (0, 255, 120)),
        (f"FPS inst : {inst_fps:6.2f}",  (0, 210, 255)),
        (f"Frame    : {frame_idx:4d} / {total_frames}", (200, 200, 200)),
        (f"Active   : {active_count:4d}",               (0, 255, 255)),
        (f"Total IDs: {total_unique:4d}",               (255, 200, 0)),
    ]

    for text, color in lines:
        cv2.putText(frame, text, (tx, ty), font, 0.60, color, 2)
        ty += lh


for image_name in image_files:

    image_path = os.path.join(SEQUENCE_FOLDER, image_name)
    frame = cv2.imread(image_path)
    if frame is None:
        continue

    frame_count += 1
    annotated_frame = frame.copy()


    if frame_count % DETECTION_INTERVAL == 0:
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            iou=IOU,
            imgsz=IMGSZ,
            verbose=False
        )
        last_results = results
    else:
        results = last_results

    if results is None:
        continue

    active_ids = set()

    if (results[0].boxes is not None and
            results[0].boxes.id is not None):

        boxes       = results[0].boxes.xyxy.cpu().numpy()
        track_ids   = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confidences):

            x1, y1, x2, y2 = map(int, box)

            dynamic_conf = 0.15 if y2 < height * 0.45 else 0.25
            if conf < dynamic_conf:
                continue

            
            track_seen_count[track_id] += 1
            if track_seen_count[track_id] < 3:
                continue

            active_ids.add(track_id)
            total_detections_ever.add(track_id)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            cv2.putText(
                annotated_frame, f"ID {track_id} {conf:.2f}",
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
            )

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            track = track_history[track_id]
            if len(track) > 0:
                px, py = track[-1]
                cx = int(0.7 * px + 0.3 * cx)
                cy = int(0.7 * py + 0.3 * cy)

            track.append((cx, cy))
            if len(track) > TAIL_LENGTH:
                track.pop(0)

            pts = list(track)
            for i in range(1, len(pts)):
                color = TAIL_COLORS[min(i, len(TAIL_COLORS) - 1)]
                cv2.line(annotated_frame, pts[i - 1], pts[i], color, 2)

            cv2.circle(annotated_frame, (cx, cy), 3, (0, 0, 255), -1)


    current_time = time.time()
    inst_fps     = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time    = current_time
    avg_fps      = frame_count / max(current_time - start_total, 1e-6)

    draw_hud(
        annotated_frame,
        avg_fps      = avg_fps,
        inst_fps     = inst_fps,
        frame_idx    = frame_count,
        total_frames = len(image_files),
        active_count = len(active_ids),
        total_unique = len(total_detections_ever)
    )


    if SHOW_WINDOW:
        cv2.namedWindow("Aerial Guardian", cv2.WINDOW_NORMAL)
        cv2.imshow("Aerial Guardian", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.write(annotated_frame)

    if frame_count % 100 == 0:
        print(f"[INFO] {frame_count}/{len(image_files)} frames | "
              f"avg FPS: {avg_fps:.2f} | active: {len(active_ids)}")


out.release()
cv2.destroyAllWindows()

final_fps = frame_count / max(time.time() - start_total, 1e-6)

print("\n===================================")
print("[INFO] Processing Finished")
print(f"[INFO] Frames     : {frame_count}")
print(f"[INFO] Avg FPS    : {final_fps:.2f}")
print(f"[INFO] Unique IDs : {len(total_detections_ever)}")
print(f"[INFO] Output     : {OUTPUT_PATH}")
print("===================================")