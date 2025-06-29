import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

def reid_single_video(video_path, model_path, output_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    tracker = DeepSort(max_age=30)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            cls_id = int(cls)

            # Track only class 2 = player and confidence above 0.4
            if cls_id == 2 and score > 0.4:
                bbox = [x1, y1, x2 - x1, y2 - y1]  # (x, y, w, h)
                detections.append((bbox, score, cls_id))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.det_class != 2:
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    reid_single_video("input/15sec_input_720p.mp4", "model/yolov8.pt", "output/result_with_ids.mp4")