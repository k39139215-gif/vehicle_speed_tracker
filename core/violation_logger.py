import os
import csv
import time
from datetime import datetime

class ViolationLogger:
    """
    Structured logger for traffic infractions. Records offenses to CSV
    with deduplication/cooldown to avoid multi-frame log spamming.
    """
    def __init__(self, log_dir="reports", filename="traffic_violations.csv", cooldown_seconds=4.0):
        self.log_dir = log_dir
        self.filename = filename
        self.filepath = os.path.join(self.log_dir, self.filename)
        self.cooldown_seconds = cooldown_seconds
        
        # Tracks last log timestamp per (track_id, violation_type)
        self._logged_events = {}
        
        # Violation counts by type
        self.summary_counts = {
            "WRONG_WAY": 0,
            "OVER_SPEEDING": 0,
            "RESTRICTED_LANE": 0,
            "LANE_STRADDLING": 0
        }
        self.total_violations_logged = 0

        self._init_csv()

    def _init_csv(self):
        os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "Frame",
                    "Vehicle_ID",
                    "Vehicle_Type",
                    "Speed_KMH",
                    "Lane",
                    "Violation_Type",
                    "Severity",
                    "Details"
                ])

    def log(self, frame_idx, vehicle, lane_info, violation_dict):
        """
        Logs an offense if not logged recently for this vehicle and violation type.
        
        Args:
            frame_idx: current video frame number
            vehicle: TrackedVehicle instance
            lane_info: dict with lane information
            violation_dict: violation detail from ViolationDetector
        """
        v_type = violation_dict["type"]
        track_id = vehicle.track_id
        now = time.time()
        
        key = (track_id, v_type)
        last_time = self._logged_events.get(key, 0.0)
        
        if now - last_time < self.cooldown_seconds:
            return False  # Suppress duplicate

        self._logged_events[key] = now
        self.summary_counts[v_type] = self.summary_counts.get(v_type, 0) + 1
        self.total_violations_logged += 1

        lane_name = lane_info.get("lane_name", "Unknown")
        speed_val = f"{vehicle.current_speed_kmh:.1f}" if vehicle.current_speed_kmh is not None else "N/A"
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                time_str,
                frame_idx,
                track_id,
                vehicle.class_name,
                speed_val,
                lane_name,
                v_type,
                violation_dict["severity"],
                violation_dict["message"]
            ])
            
        return True

    def get_summary(self):
        return {
            "total_violations": self.total_violations_logged,
            "counts_by_type": dict(self.summary_counts),
            "csv_path": os.path.abspath(self.filepath)
        }
