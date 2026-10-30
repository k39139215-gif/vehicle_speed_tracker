import os
import sys
import time
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

tracker_dir = os.path.dirname(os.path.abspath(__file__))
if tracker_dir not in sys.path:
    sys.path.append(tracker_dir)

from run_live import CAMERAS

class ViewTransformer:
    """
    Exact Roboflow 4-point Perspective Homography Transformer.
    Maps 2D pixel coordinates to real-world metric space (X, Y) in meters.
    """
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)

def start_roboflow_live_player():
    print("=" * 75)
    print("   🏎️  ROBOFLOW OFFICIAL STYLE: VEHICLE SPEED ESTIMATION & ByteTRACK")
    print("=" * 75)
    print("  -> 100% LOCAL & OFFLINE (ZERO API REQUIRED - Uses Ultralytics + Supervision)")
    print("  -> Controls:")
    print("     [N]       : NEXT Camera Angle")
    print("     [P]       : PREVIOUS Camera Angle")
    print("     [1 to 9]  : Jump directly to Camera 1 - 9")
    print("     [0]       : Jump to Camera 10")
    print("     [-]       : Jump to Camera 11")
    print("     [Z]       : Toggle Polygon Zone Overlay ON / OFF")
    print("     [T]       : Toggle Motion Traces ON / OFF")
    print("     [SPACE]   : Pause / Resume")
    print("     [Q / ESC] : Close & Exit")
    print("=" * 75)

    window_name = "ROBOFLOW SUPERVISION SPEED ESTIMATION | 100% LOCAL (NO API)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Initialize YOLOv8 vehicle detector (Classes 2:car, 3:motorcycle, 5:bus, 7:truck)
    model = YOLO("yolov8n.pt")
    
    current_cam_idx = 0
    total_cams = len(CAMERAS)

    cap = None
    fps = 25.0
    frame_w = 1280
    frame_h = 720

    byte_track = None
    polygon_zone = None
    view_transformer = None
    box_annotator = None
    label_annotator = None
    trace_annotator = None
    zone_annotator = None

    coordinates = defaultdict(lambda: deque(maxlen=int(fps)))
    smoothed_speeds = {}

    show_zone = True
    show_traces = True
    is_paused = False

    banner_text = ""
    banner_timer = 0

    def load_camera(idx):
        nonlocal cap, fps, frame_w, frame_h, byte_track, polygon_zone, view_transformer
        nonlocal box_annotator, label_annotator, trace_annotator, zone_annotator
        nonlocal coordinates, smoothed_speeds, banner_text, banner_timer

        cam = CAMERAS[idx]
        if cap is not None:
            cap.release()

        cap = cv2.VideoCapture(cam["source"])
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {cam['source']}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        if raw_w > 1280:
            scale = 1280.0 / raw_w
            frame_w = 1280
            frame_h = int(raw_h * scale)
        else:
            frame_w = raw_w
            frame_h = raw_h

        # Compute 4 road corners from ratios
        ratios = cam.get("poly_ratio", [[0.15, 0.35], [0.85, 0.35], [0.95, 0.95], [0.05, 0.95]])
        source_pts = np.array([[int(r[0] * frame_w), int(r[1] * frame_h)] for r in ratios], dtype=np.float32)

        real_len = float(cam.get("real_length", 50.0))
        real_wid = float(cam.get("real_width", 15.0))

        target_pts = np.array([
            [0.0, 0.0],
            [real_wid, 0.0],
            [real_wid, real_len],
            [0.0, real_len]
        ], dtype=np.float32)

        view_transformer = ViewTransformer(source=source_pts, target=target_pts)
        polygon_zone = sv.PolygonZone(polygon=source_pts.astype(int))

        byte_track = sv.ByteTrack(track_activation_threshold=0.28)

        # Supervision Annotators (Matching exact Roboflow video styling)
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(
            text_scale=0.52,
            text_thickness=2,
            text_position=sv.Position.BOTTOM_CENTER
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=int(fps * 1.5),
            position=sv.Position.BOTTOM_CENTER
        )
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=polygon_zone,
            color=sv.Color(r=0, g=220, b=255),
            thickness=2
        )

        coordinates = defaultdict(lambda: deque(maxlen=int(fps)))
        smoothed_speeds = {}

        banner_text = f"CAM {idx+1}/{total_cams}: {cam['title'].upper()}"
        banner_timer = int(fps * 2.5)
        print(f"[SWITCH] Loaded Camera {idx+1}/{total_cams}: {cam['title']} (Limit: {cam['speed_limit']} km/h)")
        return True

    # Load initial camera
    load_camera(current_cam_idx)

    frame_idx = 0
    prev_time = time.time()
    display_frame = None

    while True:
        curr_time = time.time()

        if not is_paused and cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                coordinates.clear()
                smoothed_speeds.clear()
                byte_track = sv.ByteTrack(track_activation_threshold=0.28)
                continue

            frame_idx += 1
            if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
                frame = cv2.resize(frame, (frame_w, frame_h))

            # 1. Run YOLO vehicle detection
            results = model(frame, conf=0.28, iou=0.5, classes=[2, 3, 5, 7], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # 2. Filter detections inside active road polygon zone
            in_zone = polygon_zone.trigger(detections)
            detections = detections[in_zone]

            # 3. Update ByteTrack
            detections = byte_track.update_with_detections(detections=detections)

            # 4. Transform bottom-center ground points into real-world metric space
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            metric_points = view_transformer.transform_points(points=points)

            for tid, (mx, my) in zip(detections.tracker_id, metric_points):
                coordinates[tid].append((frame_idx / fps, mx, my))

            # 5. Roboflow Speed Calculation (Distance / Time)
            labels = []
            speed_limit = CAMERAS[current_cam_idx]["speed_limit"]

            for tid in detections.tracker_id:
                hist = coordinates[tid]
                if len(hist) < fps / 2:
                    labels.append(f"#{tid}")
                else:
                    # Metric distance traversed across the time window
                    dt = hist[-1][0] - hist[0][0]
                    dx = hist[-1][1] - hist[0][1]
                    dy = hist[-1][2] - hist[0][2]
                    dist_meters = float(np.sqrt(dx * dx + dy * dy))
                    raw_speed_kmh = (dist_meters / max(dt, 0.001)) * 3.6

                    # Physical inertia smoothing (prevents abrupt jumping)
                    if tid not in smoothed_speeds:
                        smoothed_speeds[tid] = raw_speed_kmh
                    else:
                        max_change = 7.2 * (1.0 / fps)  # 2.0 m/s^2 acceleration limit
                        delta = np.clip(raw_speed_kmh - smoothed_speeds[tid], -max_change, max_change)
                        smoothed_speeds[tid] += delta

                    curr_spd = int(round(smoothed_speeds[tid]))
                    labels.append(f"#{tid} {curr_spd} km/h")

            # 6. Render Supervision Annotations (Exact same visual styling as video)
            annotated = frame.copy()
            if show_zone and zone_annotator is not None:
                annotated = zone_annotator.annotate(scene=annotated)
            if show_traces and trace_annotator is not None:
                annotated = trace_annotator.annotate(scene=annotated, detections=detections)
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            # 7. Notification Banner on Camera Change
            if banner_timer > 0:
                banner_timer -= 1
                h_d, w_d = annotated.shape[:2]
                bw = min(w_d - 40, 780)
                bx = (w_d - bw) // 2
                cv2.rectangle(annotated, (bx, 15), (bx + bw, 52), (15, 20, 25), -1)
                cv2.rectangle(annotated, (bx, 15), (bx + bw, 52), (0, 230, 255), 2)
                cv2.putText(annotated, banner_text[:68], (bx + 15, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 2, cv2.LINE_AA)

            display_frame = annotated

        if display_frame is not None:
            cv2.imshow(window_name, display_frame)

        target_delay = 1.0 / max(fps, 10.0)
        elapsed_loop = time.time() - curr_time
        wait_ms = max(1, int((target_delay - elapsed_loop) * 1000)) if not is_paused else 25

        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            print("\n[EXIT] Closing Roboflow Live Player.")
            break
        elif key == ord('n') or key == ord('N') or key == 83:  # N or Right Arrow
            current_cam_idx = (current_cam_idx + 1) % total_cams
            load_camera(current_cam_idx)
        elif key == ord('p') or key == ord('P') or key == 81:  # P or Left Arrow
            current_cam_idx = (current_cam_idx - 1 + total_cams) % total_cams
            load_camera(current_cam_idx)
        elif ord('1') <= key <= ord('9'):
            target_idx = key - ord('1')
            if target_idx < total_cams:
                current_cam_idx = target_idx
                load_camera(current_cam_idx)
        elif key == ord('0'):
            if total_cams >= 10:
                current_cam_idx = 9
                load_camera(current_cam_idx)
        elif key == ord('-'):
            if total_cams >= 11:
                current_cam_idx = 10
                load_camera(current_cam_idx)
        elif key == 32:  # SPACE
            is_paused = not is_paused
            print(f"[ACTION] Playback {'PAUSED' if is_paused else 'RESUMED'}")
        elif key == ord('z') or key == ord('Z'):
            show_zone = not show_zone
            print(f"[ACTION] Polygon Zone: {'ON' if show_zone else 'OFF'}")
        elif key == ord('t') or key == ord('T'):
            show_traces = not show_traces
            print(f"[ACTION] Motion Traces: {'ON' if show_traces else 'OFF'}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_roboflow_live_player()
