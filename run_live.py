import os
import sys
import time
import cv2
import numpy as np

# Ensure project root in sys.path
tracker_dir = os.path.dirname(os.path.abspath(__file__))
if tracker_dir not in sys.path:
    sys.path.append(tracker_dir)

from core.vehicle_detector import VehicleDetector
from core.vehicle_tracker import VehicleTracker, TrackedVehicle
from core.perspective_transformer import PerspectiveTransformer
from core.lane_manager import LaneManager
from core.speed_calculator import SpeedCalculator
from core.radar_evaluator import RadarGroundTruthEvaluator
from core.violation_detector import ViolationDetector
from core.violation_logger import ViolationLogger
from ui.traffic_hud import TrafficHUD

hw_720_path = os.path.join(tracker_dir, "data", "highway_traffic_720p.mp4")
hw_default_path = hw_720_path if os.path.exists(hw_720_path) else os.path.join(tracker_dir, "data", "highway_traffic.mp4")

CAMERAS = [
    {
        "id": "cam01_highway_4k",
        "title": "Highway Traffic (Dual-Carriageway Speed Benchmark)",
        "source": hw_default_path,
        "speed_limit": 100.0,
        "real_length": 82.0,
        "real_width": 24.0,
        "poly_ratio": [[0.28, 0.36], [0.72, 0.36], [0.96, 0.98], [0.04, 0.98]]
    },
    {
        "id": "cam02_distant_freeway",
        "title": "AI City Angle 1: Distant Freeway CCTV (320x240 Low-Res)",
        "source": os.path.join(tracker_dir, "data", "aicity_challenge", "angle1_distant_highway_cctv.mp4"),
        "speed_limit": 90.0,
        "real_length": 50.0,
        "real_width": 16.0,
        "poly_ratio": [[0.25, 0.20], [0.75, 0.20], [0.95, 0.95], [0.05, 0.95]]
    },
    {
        "id": "cam03_night_intersection",
        "title": "AI City Angle 2: Pitch-Black Nocturnal CCTV (02:38 AM Glare)",
        "source": os.path.join(tracker_dir, "data", "aicity_challenge", "angle2_dawn_lowlight_intersection.mp4"),
        "speed_limit": 50.0,
        "real_length": 45.0,
        "real_width": 14.0,
        "poly_ratio": [[0.20, 0.35], [0.80, 0.35], [0.95, 0.95], [0.05, 0.95]]
    },
    {
        "id": "cam04_roundabout",
        "title": "AI City Angle 3: Complex Multi-Lane Roundabout (Curved Turns)",
        "source": os.path.join(tracker_dir, "data", "aicity_challenge", "angle3_congested_roundabout.mp4"),
        "speed_limit": 45.0,
        "real_length": 35.0,
        "real_width": 15.0,
        "poly_ratio": [[0.15, 0.30], [0.85, 0.30], [0.95, 0.95], [0.05, 0.95]]
    },
    {
        "id": "cam05_oblique_pole",
        "title": "AI City Angle 4: Oblique Pole View (Trucks + Scooters + Cars)",
        "source": os.path.join(tracker_dir, "data", "aicity_challenge", "angle4_acute_oblique_anomaly.mp4"),
        "speed_limit": 60.0,
        "real_length": 35.0,
        "real_width": 14.0,
        "poly_ratio": [[0.25, 0.35], [0.75, 0.35], [0.95, 0.95], [0.05, 0.95]]
    },
    {
        "id": "cam06_roadside_occlusion",
        "title": "AI City Angle 5: Roadside View (Pedestrians + Mixed Traffic)",
        "source": os.path.join(tracker_dir, "data", "aicity_challenge", "angle5_close_quarters_occlusion.mp4"),
        "speed_limit": 50.0,
        "real_length": 30.0,
        "real_width": 12.0,
        "poly_ratio": [[0.20, 0.30], [0.80, 0.30], [0.95, 0.95], [0.05, 0.95]]
    },
    {
        "id": "cam07_tunnel_corridor",
        "title": "[NEW] Angle 1: Subterranean Roadway Tunnel (Sodium Glare)",
        "source": os.path.join(tracker_dir, "data", "new_hard_angles", "video1_tunnel_surveillance.mp4"),
        "speed_limit": 60.0,
        "real_length": 24.0,
        "real_width": 10.0,
        "poly_ratio": [[0.30, 0.40], [0.70, 0.40], [0.95, 0.95], [0.05, 0.95]]
    },
    {
        "id": "cam08_night_mast",
        "title": "[NEW] Angle 2: Elevated 02:00 AM Night-Mast Intersection",
        "source": os.path.join(tracker_dir, "data", "new_hard_angles", "video2_elevated_night_intersection.mp4"),
        "speed_limit": 50.0,
        "real_length": 45.0,
        "real_width": 14.0,
        "poly_ratio": [[0.15, 0.25], [0.85, 0.25], [0.95, 0.95], [0.05, 0.95]]
    },
    {
        "id": "cam09_blizzard_snow",
        "title": "[NEW] Angle 3: Severe Blizzard & Snowstorm Highway (Whiteout)",
        "source": os.path.join(tracker_dir, "data", "new_hard_angles", "video3_blizzard_snow_roadway.mp4"),
        "speed_limit": 45.0,
        "real_length": 55.0,
        "real_width": 15.0,
        "poly_ratio": [[0.25, 0.30], [0.75, 0.30], [0.95, 0.95], [0.05, 0.95]]
    },
    {
        "id": "cam10_zebra_crossing",
        "title": "[NEW] Angle 4: Multi-Branch Zebra Crossing & T-Junction",
        "source": os.path.join(tracker_dir, "data", "new_hard_angles", "video4_zebra_crossing_tjunction.mp4"),
        "speed_limit": 40.0,
        "real_length": 25.0,
        "real_width": 12.0,
        "poly_ratio": [[0.20, 0.35], [0.80, 0.35], [0.95, 0.95], [0.05, 0.95]]
    },
    {
        "id": "cam11_curved_overpass",
        "title": "[NEW] Angle 5: Acute Curved Highway Overpass & On-Ramp",
        "source": os.path.join(tracker_dir, "data", "new_hard_angles", "video5_curved_highway_overpass.mp4"),
        "speed_limit": 70.0,
        "real_length": 45.0,
        "real_width": 14.0,
        "poly_ratio": [[0.25, 0.35], [0.75, 0.35], [0.95, 0.95], [0.05, 0.95]]
    }
]

def calibrate_interactive(cap, window_name):
    """Allows user to click 4 road corners interactively."""
    ret, frame = cap.read()
    if not ret:
        return None

    clone = frame.copy()
    clicked = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append([x, y])
            cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
            if len(clicked) > 1:
                cv2.line(clone, tuple(clicked[-2]), tuple(clicked[-1]), (0, 255, 255), 2)
            if len(clicked) == 4:
                cv2.line(clone, tuple(clicked[3]), tuple(clicked[0]), (0, 255, 255), 2)
            cv2.imshow(window_name, clone)

    cv2.setMouseCallback(window_name, on_mouse)
    print("\n[CALIBRATION] Click 4 road corners: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    print("              Press SPACE or ENTER to confirm, or 'R' to reset.\n")

    while True:
        # Show prompt banner
        h_c, w_c = clone.shape[:2]
        cv2.putText(clone, "CALIBRATION: Click 4 Road Corners (TL -> TR -> BR -> BL) | SPACE to save | R to reset",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow(window_name, clone)
        k = cv2.waitKey(20) & 0xFF
        if k == 13 or k == 32:  # Enter or Space
            if len(clicked) == 4:
                break
        elif k == ord('r') or k == ord('R'):
            clicked = []
            clone = frame.copy()
        elif k == 27 or k == ord('q'):
            break

    cv2.setMouseCallback(window_name, lambda *args: None)
    if len(clicked) == 4:
        return clicked
    return None

def start_direct_gui_player():
    print("=" * 75)
    print("   🏎️  AI VEHICLE SPEED TRACKER & TRAFFIC ENFORCEMENT - DIRECT LIVE GUI")
    print("=" * 75)
    print("  -> Opening Video Window Directly on Screen...")
    print("  -> Controls inside Video Window:")
    print("     [N]       : Switch to NEXT camera angle")
    print("     [P]       : Switch to PREVIOUS camera angle")
    print("     [1 to 9]  : Jump directly to Camera 1 through 9")
    print("     [0]       : Jump to Camera 10")
    print("     [-]       : Jump to Camera 11")
    print("     [SPACE]   : Pause / Resume playback")
    print("     [L]       : Toggle Multi-Lane Corridors ON / OFF")
    print("     [C]       : Interactive Road Calibration (Mouse Clicks)")
    print("     [Q / ESC] : Close and Exit")
    print("=" * 75)

    window_name = "AI TRAFFIC ENFORCEMENT & SPEED RADAR | LIVE DIRECT PLAYER"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Initialize YOLOv8 Detector with imgsz=480 for rock-solid detection & ByteTrack accuracy
    detector = VehicleDetector(conf_thresh=0.35, imgsz=480)
    hud = TrafficHUD()

    # Warm up YOLO ByteTrack model with dummy frame so video playback starts instantly
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    _ = detector.track(dummy)

    current_cam_idx = 0
    total_cams = len(CAMERAS)

    cap = None
    tracker = None
    transformer = None
    lane_manager = None
    speed_calc = None
    evaluator = None
    violation_detector = None
    violation_logger = None

    fps = 30.0
    frame_w = 1280
    frame_h = 720
    is_paused = False
    show_lanes = False  # Keep road clean by default (No distracting polygon overlays)
    banner_text = ""
    banner_timer = 0

    active_tracks = {}

    def load_camera(idx):
        nonlocal cap, tracker, transformer, lane_manager, speed_calc, evaluator, violation_detector, violation_logger
        nonlocal fps, frame_w, frame_h, banner_text, banner_timer, active_tracks

        active_tracks = {}
        detector.reset_tracker()
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

        # Standardize frame resolution to 720p max to guarantee 30+ FPS with zero lag
        if raw_w > 1280:
            scale = 1280.0 / raw_w
            frame_w = 1280
            frame_h = int(raw_h * scale)
        else:
            frame_w = raw_w
            frame_h = raw_h

        # Calculate road polygon corners from ratios
        ratios = cam.get("poly_ratio", [[0.15, 0.35], [0.85, 0.35], [0.95, 0.95], [0.05, 0.95]])
        poly_pts = [[int(r[0] * frame_w), int(r[1] * frame_h)] for r in ratios]

        real_len = cam.get("real_length", 55.0)
        real_wid = cam.get("real_width", 18.0)

        # Initialize tracking and speed engines
        transformer = PerspectiveTransformer(source_polygon=poly_pts, real_length_meters=real_len, real_width_meters=real_wid)
        lane_manager = LaneManager(source_polygon=poly_pts, num_lanes=3)
        speed_calc = SpeedCalculator(transformer, fps=fps, speed_limit_kmh=cam["speed_limit"],
                                     min_history_frames=8, window_size=12, max_accel_kmh_s=7.2)
        tracker = VehicleTracker(max_distance_threshold=65.0, max_lost_frames=12)
        evaluator = RadarGroundTruthEvaluator()
        violation_detector = ViolationDetector(min_motion_px=10.0)
        violation_logger = ViolationLogger(log_dir=os.path.join(tracker_dir, "reports"))

        banner_text = f"CAM {idx+1}/{total_cams}: {cam['title'].upper()}"
        banner_timer = int(fps * 2.0)
        print(f"[SWITCH] Loaded Camera {idx+1}/{total_cams}: {cam['title']}")
        return True

    # Load initial camera (Highway Traffic)
    load_camera(current_cam_idx)

    prev_time = time.time()
    frame_idx = 0
    display_frame = None

    while True:
        curr_time = time.time()

        if not is_paused and cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop video seamlessly with clean track history
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                active_tracks.clear()
                detector.reset_tracker()
                continue

            frame_idx += 1
            effective_fps = 1.0 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time

            # Resize to standard frame_w, frame_h if needed
            if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
                frame = cv2.resize(frame, (frame_w, frame_h))

            timestamp = frame_idx / fps

            # 1. Kalman-filtered ByteTrack vehicle tracking (Smooth, zero drift)
            tracked_items = detector.track(frame, persist=True)
            current_ids = set()
            for item in tracked_items:
                tid = item['track_id']
                current_ids.add(tid)
                if tid in active_tracks:
                    active_tracks[tid].update(item['bbox'], item['ground_point'], timestamp)
                    active_tracks[tid].lost_frames = 0
                else:
                    active_tracks[tid] = TrackedVehicle(tid, item['bbox'], item['ground_point'], item['class_name'], timestamp)

            # Instantly remove vehicles that left scene (no wandering phantom boxes)
            lost_ids = []
            for tid, trk in active_tracks.items():
                if tid not in current_ids:
                    trk.lost_frames += 1
                    px, py = trk.history[-1][:2]
                    near_edge = (px < 20 or px > frame_w - 20 or py < 20 or py > frame_h - 20)
                    if trk.lost_frames > (2 if near_edge else 12):
                        lost_ids.append(tid)
            for tid in lost_ids:
                del active_tracks[tid]

            # 2. Compute real-time vehicle speed in km/h
            for tid, trk in active_tracks.items():
                speed_calc.compute_speed(trk, active_tracks)

            # 3. Render High-Contrast HUD (Clean road, bold speed tags)
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
                frame_idx=frame_idx,
                speed_limit_kmh=CAMERAS[current_cam_idx]["speed_limit"]
            )

            # 4. Top Camera Notification Banner (on camera change)
            if banner_timer > 0:
                banner_timer -= 1
                h_disp, w_disp = display_frame.shape[:2]
                box_w = min(w_disp - 40, 750)
                bx = (w_disp - box_w) // 2
                cv2.rectangle(display_frame, (bx, 58), (bx + box_w, 96), (10, 14, 20), -1)
                cv2.rectangle(display_frame, (bx, 58), (bx + box_w, 96), (0, 230, 255), 2)
                cv2.putText(display_frame, banner_text[:65], (bx + 15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 2, cv2.LINE_AA)

        if display_frame is not None:
            cv2.imshow(window_name, display_frame)

        # Real-time pacing: sync playback speed to natural 1.0x video FPS
        target_delay = 1.0 / max(fps, 10.0)
        elapsed_loop = time.time() - curr_time
        wait_ms = max(1, int((target_delay - elapsed_loop) * 1000)) if not is_paused else 25

        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            print("\n[EXIT] Closing Live Window.")
            break
        elif key == ord('n') or key == ord('N') or key == 83:  # N or Right Arrow -> Next Cam
            current_cam_idx = (current_cam_idx + 1) % total_cams
            load_camera(current_cam_idx)
        elif key == ord('p') or key == ord('P') or key == 81:  # P or Left Arrow -> Prev Cam
            current_cam_idx = (current_cam_idx - 1 + total_cams) % total_cams
            load_camera(current_cam_idx)
        elif ord('1') <= key <= ord('9'):  # Direct jump 1 to 9
            target_idx = key - ord('1')
            if target_idx < total_cams:
                current_cam_idx = target_idx
                load_camera(current_cam_idx)
        elif key == ord('0'):  # Jump to Cam 10
            if total_cams >= 10:
                current_cam_idx = 9
                load_camera(current_cam_idx)
        elif key == ord('-'):  # Jump to Cam 11
            if total_cams >= 11:
                current_cam_idx = 10
                load_camera(current_cam_idx)
        elif key == 32:  # SPACE -> Pause / Resume
            is_paused = not is_paused
            print(f"[ACTION] Playback {'PAUSED' if is_paused else 'RESUMED'}")
        elif key == ord('l') or key == ord('L'):  # L -> Toggle Lanes
            show_lanes = not show_lanes
            print(f"[ACTION] Multi-Lane Corridors: {'ON' if show_lanes else 'OFF'}")
        elif key == ord('c') or key == ord('C'):  # C -> Interactive Calibration
            new_pts = calibrate_interactive(cap, window_name)
            if new_pts:
                transformer = PerspectiveTransformer(source_polygon=new_pts, real_length_meters=80.0, real_width_meters=14.0)
                lane_manager.reconfigure(new_pts)
                speed_calc = SpeedCalculator(transformer, fps=fps, speed_limit_kmh=CAMERAS[current_cam_idx]["speed_limit"])
                print("[ACTION] Road Geometry successfully re-calibrated.")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_direct_gui_player()
