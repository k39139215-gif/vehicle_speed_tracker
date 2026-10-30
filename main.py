import cv2
import time
import os
import argparse
import numpy as np

from utils.video_downloader import download_benchmark_video
from core.perspective_transformer import PerspectiveTransformer
from core.vehicle_detector import VehicleDetector
from core.vehicle_tracker import VehicleTracker
from core.speed_calculator import SpeedCalculator
from core.radar_evaluator import RadarGroundTruthEvaluator
from core.lane_manager import LaneManager
from core.violation_detector import ViolationDetector
from core.violation_logger import ViolationLogger
from ui.traffic_hud import TrafficHUD

def calibrate_interactive(cap):
    """
    Allows the user to click 4 points on the road to dynamically
    calibrate any custom traffic video.
    """
    ret, frame = cap.read()
    if not ret:
        return None

    clone = frame.copy()
    clicked_points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append([x, y])
            cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
            if len(clicked_points) > 1:
                cv2.line(clone, tuple(clicked_points[-2]), tuple(clicked_points[-1]), (0, 255, 255), 2)
            if len(clicked_points) == 4:
                cv2.line(clone, tuple(clicked_points[3]), tuple(clicked_points[0]), (0, 255, 255), 2)
            cv2.imshow("CALIBRATION (Click 4 Road Corners: TL, TR, BR, BL)", clone)

    cv2.namedWindow("CALIBRATION (Click 4 Road Corners: TL, TR, BR, BL)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("CALIBRATION (Click 4 Road Corners: TL, TR, BR, BL)", on_mouse)

    print("\n[CALIBRATION MODE]")
    print("  Click 4 points on the road: 1. Top-Left, 2. Top-Right, 3. Bottom-Right, 4. Bottom-Left")
    print("  Press Enter or Space when finished, or 'r' to reset.\n")

    while True:
        cv2.imshow("CALIBRATION (Click 4 Road Corners: TL, TR, BR, BL)", clone)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 or key == 32:  # Enter or Space
            if len(clicked_points) == 4:
                break
        elif key == ord('r') or key == ord('R'):
            clicked_points = []
            clone = frame.copy()
        elif key == 27 or key == ord('q'):
            break

    cv2.destroyWindow("CALIBRATION (Click 4 Road Corners: TL, TR, BR, BL)")
    if len(clicked_points) == 4:
        return clicked_points
    return None

def main():
    parser = argparse.ArgumentParser(description="Smart Multi-Lane Traffic Enforcement & Speed Radar")
    parser.add_argument("--source", default=None, help="Path to traffic video (defaults to benchmark video)")
    parser.add_argument("--speed-limit", type=float, default=80.0, help="Default speed limit in km/h (default: 80)")
    parser.add_argument("--calibrate", action="store_true", help="Launch interactive 4-point calibration")
    args = parser.parse_args()

    print("=" * 75)
    print("  INTELLIGENT TRAFFIC ENFORCEMENT & MULTI-LANE SPEED RADAR SYSTEM")
    print("=" * 75)

    # Resolve Video Source (Local Benchmark Video, No repeated downloads)
    video_path = args.source
    if video_path is None or not os.path.exists(video_path):
        video_path = os.path.join(os.path.dirname(__file__), "data", "highway_traffic.mp4")
        if not os.path.exists(video_path):
            download_benchmark_video(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[FATAL] Unable to open video source at: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video loaded: {video_path} ({width}x{height} @ {fps:.1f} FPS)")

    # 4-Point Homography Road Calibration
    custom_points = None
    if args.calibrate:
        custom_points = calibrate_interactive(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize Core Engines
    transformer = PerspectiveTransformer(source_polygon=custom_points, real_length_meters=80.0, real_width_meters=14.0)
    lane_manager = LaneManager(source_polygon=custom_points, num_lanes=3)
    violation_detector = ViolationDetector(min_motion_px=12.0)
    violation_logger = ViolationLogger(log_dir=os.path.join(os.path.dirname(__file__), "reports"))
    
    detector = VehicleDetector(conf_thresh=0.28)
    tracker = VehicleTracker(max_distance_threshold=70.0, max_lost_frames=15)
    speed_calc = SpeedCalculator(transformer, fps=fps, speed_limit_kmh=args.speed_limit, min_history_frames=8)
    evaluator = RadarGroundTruthEvaluator()
    hud = TrafficHUD()

    window_name = "AI TRAFFIC ENFORCEMENT | Multi-Lane & Speed Radar"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 720)

    prev_time = time.time()
    is_paused = False
    show_lanes = True
    frame_idx = 0

    print("\n[ACTIVE KEYBOARD CONTROLS]")
    print("  [SPACE] : Pause / Resume Video")
    print("  [L]     : Toggle Multi-Lane Corridors ON/OFF")
    print("  [C]     : Re-calibrate Road Polygon (Interactive Clicks)")
    print("  [Q/ESC] : Exit Application & Print Violation Report\n")

    display_frame = None

    while cap.isOpened():
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                # Loop video for continuous evaluation
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx += 1
            curr_time = time.time()
            effective_fps = 1.0 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time

            # 1. Detect Vehicles (YOLOv8)
            detections = detector.detect(frame)

            # 2. Multi-Object Tracking (Persistent IDs)
            timestamp = frame_idx / fps
            active_tracks = tracker.update(detections, timestamp=timestamp)

            # 3. Compute Speeds using Homography & Distance over Time
            for tid, trk in active_tracks.items():
                speed_calc.compute_speed(trk)

            # 4. Render Tactical Traffic Enforcement HUD
            display_frame = hud.render(
                frame,
                tracked_vehicles=active_tracks,
                transformer=transformer,
                lane_manager=lane_manager,
                violation_detector=violation_detector,
                violation_logger=violation_logger,
                evaluator=evaluator,
                fps=effective_fps,
                show_lanes=show_lanes,
                frame_idx=frame_idx
            )

        if display_frame is not None:
            cv2.imshow(window_name, display_frame)

        # Handle Keyboard Events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 32:  # Space
            is_paused = not is_paused
            print(f"[ACTION] Video {'PAUSED' if is_paused else 'RESUMED'}")
        elif key == ord('l') or key == ord('L'):
            show_lanes = not show_lanes
            print(f"[ACTION] Multi-Lane Corridors Display: {'ON' if show_lanes else 'OFF'}")
        elif key == ord('c') or key == ord('C'):
            new_pts = calibrate_interactive(cap)
            if new_pts:
                transformer = PerspectiveTransformer(source_polygon=new_pts, real_length_meters=80.0, real_width_meters=14.0)
                lane_manager.reconfigure(new_pts)
                speed_calc = SpeedCalculator(transformer, fps=fps, speed_limit_kmh=args.speed_limit)
                print("[ACTION] Road & Multi-Lane Geometry successfully re-calibrated.")

    cap.release()
    cv2.destroyAllWindows()
    
    # Print Final Summary Report
    summary = violation_logger.get_summary()
    print("\n" + "=" * 75)
    print("  TRAFFIC ENFORCEMENT SESSION SUMMARY")
    print("=" * 75)
    print(f"  Total Violations Logged: {summary['total_violations']}")
    for v_type, count in summary['counts_by_type'].items():
        print(f"    - {v_type:20s}: {count} incidents")
    print(f"  Official CSV Report File: {summary['csv_path']}")
    print("=" * 75)

if __name__ == "__main__":
    main()
