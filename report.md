# Player Re-Identification – Report

## Approach and Methodology

The goal of this project was to identify and track individual players from a single sports video feed using object detection and tracking techniques. I used the following approach:

1. **YOLOv8 for Detection**  
   - YOLOv8 was used to detect players (class: person) in each video frame.
   - It provides fast and accurate bounding box predictions.

2. **Deep SORT for Tracking**  
   - Deep SORT was integrated to assign consistent IDs to detected players across frames.
   - It combines appearance features with Kalman filter prediction and Hungarian matching.

3. **Annotation**  
   - Bounding boxes were drawn with player IDs to visualize consistent tracking.
   - Goalkeeper and ball detection (as additional features) were highlighted using different colors.

## Techniques Tried and Outcomes

- **YOLOv8** outperformed other detectors in terms of speed and accuracy.
- **Deep SORT** handled occlusions and re-identification well during short player disappearances.
- Attempted a simple ball detection, but tracking the ball accurately over time requires a dedicated method (due to its fast and small nature).

## Challenges Encountered

- **Ball Tracking:**  
  Difficult due to its small size, speed, and frequent occlusion.

- **Model Size Limits on GitHub:**  
  Input/output videos and YOLO weights were too large for direct upload. I handled this by sharing them via a Google Drive link (see `README.md`).

- **Camera Motion:**  
  When the camera moves or zooms, it sometimes affects the consistency of tracking.

## What Remains and Future Work

If I had more time or resources, I would:

- Implement **Re-ID feature vectors** to improve player matching even after long occlusion.
- Build a **custom classifier** to detect goalkeeper by jersey color.
- Integrate **optical flow** or homography for better ball trajectory estimation.
- Try **multi-camera player re-identification** across different viewpoints.

---
