import cv2
import config
import time

from tracker.sort import SORT
from tracker.reid_module import ReIDModule
from utils.visualisation import draw_tracks
from ultralytics import YOLO

# Initialise modules
tracker = SORT(config.SORT_IOU_THRESHOLD, config.SORT_MIN_HITS) if config.USE_KALMAN else None
reid = ReIDModule(config.REID_SIM_THRESHOLD) if config.USE_REID else None

# Load the YOLO model
model = YOLO(config.YOLO_MODEL_VERSION)

# Open the video file
cap = cv2.VideoCapture(config.VIDEO_PATH)
out = None
if config.SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
detections = []
start_time = time.time()  # Start timer

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break

    if frame_count % config.FRAME_SKIP == 0:
        # Run YOLO inference on the frame
        results = model(source=frame, conf=config.YOLO_CONF_THRESHOLD, iou=config.YOLO_IOU_THRESHOLD)
        if (not config.USE_KALMAN) & (not config.USE_REID):
            frame = results[0].plot(conf=False)

        # Get the detection box (for Kalman filter update)
        detections = []
        for box in results[0].boxes:
            x, y, w, h = map(int, box.xywh[0])
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]
            detections.append({'bbox': (x - w // 2, y - h // 2, w, h), 'label': cls_name, 'cls_id': cls_id})

    if tracker:
        tracks = tracker.update(detections, frame=frame, reid=reid)
        # Draw the tracking results on the frame
        draw_tracks(frame, tracks)

    frame_count += 1

    # display/save
    if config.DISPLAY:
        cv2.imshow("Tracking", frame)  # Display the annotated frame
    if config.SAVE_VIDEO:
        out.write(frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

end_time = time.time()  # End timer
print(f"Total tracking time: {end_time - start_time:.2f} seconds")

# Release the video capture object and close the display window
cap.release()
if config.SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()
