import os
import json
import cv2
import time

class BenchmarkMemory:
    """
    Persistent Memory Database for benchmark evaluations.
    Preserves full video characteristics, trajectories, audio spectra,
    and keyframe snapshots while purging raw heavy video files from disk.
    """
    def __init__(self, memory_dir="reports", keyframe_subfolder="keyframes"):
        self.memory_dir = memory_dir
        self.keyframe_dir = os.path.join(self.memory_dir, keyframe_subfolder)
        self.json_path = os.path.join(self.memory_dir, "benchmark_memory.json")
        
        os.makedirs(self.keyframe_dir, exist_ok=True)
        self.database = {}
        self._load()

    def _load(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.database = json.load(f)
            except Exception:
                self.database = {}

    def save(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.database, f, indent=2)

    def record_run(self, video_id, vehicle_model, visual_result, audio_result, raw_video_path=None, raw_audio_path=None):
        """
        Stores complete execution telemetry into memory and deletes heavy video files.
        """
        keyframe_path = None
        if visual_result.get("sample_keyframe") is not None:
            kf_filename = f"{video_id}_keyframe.jpg"
            keyframe_path = os.path.join(self.keyframe_dir, kf_filename)
            cv2.imwrite(keyframe_path, visual_result["sample_keyframe"])

        # Sensor Fusion: 65% Visual + 35% Acoustic Doppler
        v_vis = visual_result.get("visual_speed_kmh", 0.0)
        v_aud = audio_result.get("speed_kmh", 0.0)
        if v_vis > 0 and v_aud > 0:
            fused_speed = round(0.65 * v_vis + 0.35 * v_aud, 1)
        elif v_vis > 0:
            fused_speed = v_vis
        else:
            fused_speed = v_aud

        entry = {
            "video_id": video_id,
            "vehicle_model": vehicle_model,
            "timestamp_processed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "visual": {
                "speed_kmh": v_vis,
                "cpa_frame": visual_result.get("cpa_frame", 0),
                "cpa_timestamp": visual_result.get("cpa_timestamp", 0.0),
                "frames_tracked": visual_result.get("frames_tracked", 0),
                "trajectory_points": len(visual_result.get("trajectory", []))
            },
            "acoustic": {
                "speed_kmh": v_aud,
                "cpa_timestamp": audio_result.get("cpa_timestamp", 0.0),
                "f_approach": audio_result.get("f_approach", 0.0),
                "f_recede": audio_result.get("f_recede", 0.0),
                "snr_db": audio_result.get("snr_db", 0.0),
                "confidence": audio_result.get("confidence", 0.0)
            },
            "fused_speed_kmh": fused_speed,
            "keyframe_image": os.path.abspath(keyframe_path) if keyframe_path else None
        }

        self.database[video_id] = entry
        self.save()

        # DISK SAFETY: Purge raw heavy video file immediately
        if raw_video_path and os.path.exists(raw_video_path):
            try:
                os.remove(raw_video_path)
                print(f"  [DISK CLEANUP] Purged raw video file to conserve disk: {os.path.basename(raw_video_path)}")
            except Exception as e:
                print(f"  [WARN] Failed to delete {raw_video_path}: {e}")

        # Delete raw audio file
        if raw_audio_path and os.path.exists(raw_audio_path):
            try:
                os.remove(raw_audio_path)
            except Exception:
                pass

        return entry

    def get_all_records(self):
        return self.database
