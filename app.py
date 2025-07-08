from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==== Paths to model weights and video ====
YOLO_MODEL_PATH = "best.pt"  # Replace with your YOLOv11 weights
VIDEO_INPUT_PATH = "15sec_input_720p.mp4"  # Replace with your video file
VIDEO_OUTPUT_PATH = "output_with_tracking.mp4"  # Output file

# ==== Initialize YOLOv11 model ====
print("[INFO] Loading YOLOv11 model...")
model = YOLO(YOLO_MODEL_PATH)

# ==== Debug: Check loaded model classes ====
print("[DEBUG] Model classes:", model.names)

# ==== Initialize DeepSORT tracker ====
print("[INFO] Initializing DeepSORT tracker...")
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# ==== Open the input video ====
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
if not cap.isOpened():
    print("[ERROR] Could not open video. Check the path.")
    exit()

# ==== Get video properties for output ====
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Video resolution: {frame_width}x{frame_height} @ {fps:.2f} FPS")

# ==== Define video writer for output ====
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or try 'XVID'
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

frame_count = 0

print("[INFO] Processing video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video reached.")
        break

    frame_count += 1

    # ==== Run YOLOv11 detection ====
    results = model(frame)

    detections = []

    # ==== Process YOLO detections ====
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        w, h = x2 - x1, y2 - y1
        confidence = float(box.conf[0])  # Confidence score
        class_id = int(box.cls[0])  # Class ID (e.g., player or ball)

        # Debug: print detected classes and confidence
        print(f"Frame {frame_count}: Class ID={class_id}, Confidence={confidence:.2f}, Box={x1,y1,x2,y2}")

        # ==== Filter detections ====
        # (Adjust class_id filter if needed based on model)
        if confidence > 0.91 and w > 30 and h > 30:
            detections.append(([x1, y1, w, h], confidence, 'player'))

    # ==== Update DeepSORT tracker with detections ====
    tracks = tracker.update_tracks(detections, frame=frame)

    # ==== Draw tracking results ====
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id  # Unique player ID
        ltrb = track.to_ltrb()  # Bounding box: left, top, right, bottom

        x1, y1, x2, y2 = map(int, ltrb)
        # Draw bounding box with brighter color
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f'Player {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ==== Write the processed frame to output video ====
    out.write(frame)

    # Display the frame (optional for debugging)
    cv2.imshow("Player Re-Identification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[INFO] Processing complete. Output saved to {VIDEO_OUTPUT_PATH}")
