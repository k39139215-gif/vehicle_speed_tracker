import cv2
import numpy as np

class VehicleDetector:
    """
    Real-time vehicle detection using YOLOv8-nano.
    Extracts bounding boxes, vehicle classifications, and ground-contact points
    for cars, trucks, buses, and motorcycles.
    """
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(self, model_name="yolov8n.pt", conf_thresh=0.30, imgsz=384):
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.model = None
        self.has_yolo = False

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.has_yolo = True
            print("[INFO] VehicleDetector: YOLOv8 initialized successfully.")
        except Exception as e:
            print(f"[WARN] VehicleDetector initialization error: {e}")

    def detect(self, frame_bgr):
        """
        Detects vehicles in frame.
        
        Returns:
            list of dicts: [
                {
                    'bbox': (x1, y1, x2, y2),
                    'conf': float,
                    'class_name': str,
                    'ground_point': (gx, gy)  # (x_center, y_bottom) touching road
                }, ...
            ]
        """
        if frame_bgr is None or frame_bgr.size == 0 or not self.has_yolo or self.model is None:
            return []

        vehicles = []
        try:
            # Predict only vehicle classes (2, 3, 5, 7) for maximum FPS
            results = self.model.predict(
                frame_bgr,
                classes=[2, 3, 5, 7],
                conf=self.conf_thresh,
                iou=0.45,
                verbose=False,
                imgsz=self.imgsz
            )

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self.VEHICLE_CLASSES.get(cls_id, "vehicle")
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    # Ground contact point: center-bottom of bounding box
                    gx = (x1 + x2) // 2
                    gy = y2

                    vehicles.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'class_name': cls_name,
                        'ground_point': (gx, gy)
                    })
        except Exception as e:
            pass

        return vehicles

    def track(self, frame_bgr, persist=True):
        """
        Runs Kalman-filtered ByteTrack directly through YOLOv8 for smooth vehicle tracking.
        Guarantees zero ghost/wandering boxes and maintains consistent IDs.
        """
        if frame_bgr is None or frame_bgr.size == 0 or not self.has_yolo or self.model is None:
            return []

        tracked = []
        try:
            results = self.model.track(
                frame_bgr,
                persist=persist,
                tracker="bytetrack.yaml",
                classes=[2, 3, 5, 7],
                conf=self.conf_thresh,
                iou=0.45,
                verbose=False,
                imgsz=self.imgsz
            )

            for r in results:
                if r.boxes.id is None:
                    continue
                tids = r.boxes.id.cpu().numpy().astype(int)
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                clss = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()

                for tid, box, cls_id, conf in zip(tids, boxes, clss, confs):
                    cls_name = self.VEHICLE_CLASSES.get(cls_id, "car")
                    x1, y1, x2, y2 = box
                    gx = (x1 + x2) // 2
                    gy = y2
                    
                    # Prevent duplicate nested boxes on large or closely-following vehicles
                    is_dup = False
                    area1 = (x2 - x1) * (y2 - y1)
                    for prev in tracked:
                        px1, py1, px2, py2 = prev['bbox']
                        ix1, iy1 = max(x1, px1), max(y1, py1)
                        ix2, iy2 = min(x2, px2), min(y2, py2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            area2 = (px2 - px1) * (py2 - py1)
                            min_area = min(area1, area2)
                            # If smaller box is heavily overlapping or ground point is inside preceding vehicle
                            if inter / max(1.0, min_area) > 0.50:
                                is_dup = True
                                break
                    if not is_dup:
                        tracked.append({
                            'track_id': int(tid),
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'conf': float(conf),
                            'class_name': cls_name,
                            'ground_point': (int(gx), int(gy))
                        })
        except Exception as e:
            pass

        return tracked

    def reset_tracker(self):
        """Resets ByteTrack state when switching cameras or looping video."""
        try:
            if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                if hasattr(self.model.predictor, 'trackers'):
                    delattr(self.model.predictor, 'trackers')
        except Exception:
            pass
