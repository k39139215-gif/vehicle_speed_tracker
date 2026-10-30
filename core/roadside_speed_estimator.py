import cv2
import numpy as np
import time

class RoadsideSpeedEstimator:
    """
    Estimates vehicle speed from roadside pass-by video (GoPro 1.2m height, 0.5m setback).
    Uses optical transit timing and vehicle bounding box dimensional scaling.
    """
    # Standard physical lengths (meters) for benchmark vehicles
    VEHICLE_SPECS = {
        "citroenc4picasso": {"length_m": 4.43, "width_m": 1.83, "name": "Citroen C4 Picasso"},
        "kiasportage":      {"length_m": 4.48, "width_m": 1.85, "name": "Kia Sportage"},
        "mazda3":           {"length_m": 4.46, "width_m": 1.79, "name": "Mazda 3"},
        "peugeot208":       {"length_m": 3.97, "width_m": 1.74, "name": "Peugeot 208"},
        "vwpassat":         {"length_m": 4.77, "width_m": 1.83, "name": "VW Passat B7"},
        "default":          {"length_m": 4.45, "width_m": 1.80, "name": "Standard Passenger Car"}
    }

    def __init__(self, detector, tracker, lane_distance_m=3.2, hfov_deg=86.0):
        self.detector = detector
        self.tracker = tracker
        self.lane_distance_m = float(lane_distance_m)
        self.hfov_rad = np.radians(float(hfov_deg))

    def get_vehicle_spec(self, model_hint="default"):
        hint_clean = model_hint.lower().replace(" ", "").replace("_", "")
        for key, spec in self.VEHICLE_SPECS.items():
            if key in hint_clean:
                return spec
        return self.VEHICLE_SPECS["default"]

    def process_video(self, video_path, vehicle_hint="default", max_frames=300):
        """
        Runs visual tracking across a roadside pass-by video clip.
        
        Returns:
            dict containing:
                - 'visual_speed_kmh': float
                - 'cpa_frame': int
                - 'cpa_timestamp': float
                - 'frames_tracked': int
                - 'sample_keyframe': numpy array (BGR)
                - 'trajectory': list of (x, y, t)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        spec = self.get_vehicle_spec(vehicle_hint)
        veh_length_m = spec["length_m"]

        frame_idx = 0
        detections_history = []
        best_frame = None
        max_box_area = 0

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp = frame_idx / fps

            dets = self.detector.detect(frame)
            if len(dets) > 0:
                # Pick the most prominent vehicle in the pass-by
                best_det = max(dets, key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]))
                area = (best_det["bbox"][2]-best_det["bbox"][0]) * (best_det["bbox"][3]-best_det["bbox"][1])
                
                cx = (best_det["bbox"][0] + best_det["bbox"][2]) / 2.0
                cy = (best_det["bbox"][1] + best_det["bbox"][3]) / 2.0
                bw = abs(best_det["bbox"][2] - best_det["bbox"][0])
                bh = abs(best_det["bbox"][3] - best_det["bbox"][1])

                detections_history.append({
                    "frame": frame_idx,
                    "time": timestamp,
                    "cx": cx,
                    "cy": cy,
                    "w": bw,
                    "h": bh,
                    "bbox": best_det["bbox"],
                    "conf": best_det["conf"]
                })

                if area > max_box_area:
                    max_box_area = area
                    best_frame = frame.copy()

        cap.release()

        if len(detections_history) < 4:
            return {
                "visual_speed_kmh": 0.0,
                "cpa_frame": 0,
                "cpa_timestamp": 0.0,
                "frames_tracked": 0,
                "sample_keyframe": best_frame,
                "trajectory": []
            }

        # Find Closest Point of Approach (CPA) - maximum bounding box size or center of FOV
        cpa_det = max(detections_history, key=lambda d: d["w"] * d["h"])
        cpa_frame = cpa_det["frame"]
        cpa_time = cpa_det["time"]

        # Calculate speed via optical displacement across the central observation window
        # Take samples around CPA (e.g. -6 to +6 frames)
        cpa_idx = detections_history.index(cpa_det)
        start_idx = max(0, cpa_idx - 6)
        end_idx = min(len(detections_history) - 1, cpa_idx + 6)

        d_start = detections_history[start_idx]
        d_end = detections_history[end_idx]

        dt = d_end["time"] - d_start["time"]
        if dt <= 0:
            dt = 0.1

        dx_px = abs(d_end["cx"] - d_start["cx"])
        avg_w_px = (d_start["w"] + d_end["w"] + cpa_det["w"]) / 3.0

        # Physical speed: v = (dx_px / dt) * (L_veh / W_px) * 3.6
        if avg_w_px > 10.0:
            v_ms = (dx_px / dt) * (veh_length_m / avg_w_px)
            visual_speed = float(v_ms * 3.6)
        else:
            visual_speed = 0.0

        visual_speed = float(np.clip(visual_speed, 20.0, 125.0))

        return {
            "visual_speed_kmh": round(visual_speed, 1),
            "cpa_frame": cpa_frame,
            "cpa_timestamp": round(cpa_time, 2),
            "frames_tracked": len(detections_history),
            "sample_keyframe": best_frame,
            "trajectory": [(d["cx"], d["cy"], d["time"]) for d in detections_history]
        }
