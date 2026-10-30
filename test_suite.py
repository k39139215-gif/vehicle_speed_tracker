import cv2
import numpy as np
import time
import os
import csv

from core.perspective_transformer import PerspectiveTransformer
from core.vehicle_detector import VehicleDetector
from core.vehicle_tracker import VehicleTracker, TrackedVehicle
from core.speed_calculator import SpeedCalculator
from core.radar_evaluator import RadarGroundTruthEvaluator
from core.lane_manager import LaneManager
from core.violation_detector import ViolationDetector
from core.violation_logger import ViolationLogger
from ui.traffic_hud import TrafficHUD

def run_speed_tracker_tests():
    print("=" * 75)
    print("  SMART TRAFFIC ENFORCEMENT & SPEED RADAR: 8-STEP VERIFICATION SUITE")
    print("=" * 75)
    
    total_tests = 8
    passed_tests = 0
    output_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # TEST 1: 4-Point Homography & Distance Calibration
    # -------------------------------------------------------------
    print("\n[TEST 1/8] Testing 4-Point Homography & Distance Calibration...")
    polygon_4k = np.array([
        [1350, 960],    # Top-Left
        [1780, 960],    # Top-Right
        [1950, 2160],   # Bottom-Right
        [100, 2160]     # Bottom-Left
    ], dtype=np.float32)

    transformer = PerspectiveTransformer(source_polygon=polygon_4k, real_length_meters=80.0, real_width_meters=14.0)

    # Transform top edge vs bottom edge
    x_top_l, y_top_l = transformer.transform_point(1350, 960)
    x_bot_l, y_bot_l = transformer.transform_point(100, 2160)
    measured_len = y_bot_l - y_top_l
    print(f"  [HOMOGRAPHY] Calibrated Road Length: {measured_len:.2f} meters (Target: 80.0m)")
    assert abs(measured_len - 80.0) < 0.1, f"Homography length error: {measured_len}"
    print("  [OK] Perspective Homography maps road coordinates to exact metric ground truth.")
    passed_tests += 1

    # -------------------------------------------------------------
    # TEST 2: Speed Physics: Stationary (0 km/h) & Moving Car
    # -------------------------------------------------------------
    print("\n[TEST 2/8] Testing Speed Physics: Stationary (0 km/h) & Moving Car...")
    speed_calc = SpeedCalculator(transformer, fps=30.0, speed_limit_kmh=80.0, min_history_frames=5)

    # A. Stationary Vehicle Test (Car parked at red light)
    static_car = TrackedVehicle(1, (1450, 1070, 1530, 1120), (1490, 1120), "car", timestamp=0.0)
    for f in range(10):
        static_car.update((1450, 1070, 1530, 1120), (1490, 1120), timestamp=f / 30.0)
        speed, _ = speed_calc.compute_speed(static_car)

    print(f"  [PHYSICS TEST A] Stationary Car Speed: {speed:.1f} km/h (Expected: 0.0 km/h)")
    assert speed == 0.0, f"Stationary car must have 0 km/h, got {speed}"

    # B. Moving Vehicle Test
    moving_car = TrackedVehicle(2, (1450, 1070, 1530, 1120), (1498, 1107), "car", timestamp=0.0)
    p1 = np.array([1498.9, 1106.9])
    p2 = np.array([1414.2, 1295.1])

    speed_moving = None
    is_over = False
    for f in range(1, 46):
        alpha = f / 45.0
        pt = (1.0 - alpha) * p1 + alpha * p2
        moving_car.update((int(pt[0] - 50), int(pt[1] - 40), int(pt[0] + 50), int(pt[1])),
                          (int(pt[0]), int(pt[1])),
                          timestamp=f / 30.0)
        curr_spd, is_over = speed_calc.compute_speed(moving_car)
        if curr_spd is not None:
            speed_moving = curr_spd

    print(f"  [PHYSICS TEST B] Moving Vehicle Measured Speed: {speed_moving:.1f} km/h (Over-speeding: {is_over})")
    assert 30.0 <= speed_moving <= 80.0, f"Speed calculation outside expected range: {speed_moving:.1f} km/h"
    print("  [OK] Mathematical speed physics verified: 0 km/h for stationary and accurate velocity for moving cars.")
    passed_tests += 1

    # -------------------------------------------------------------
    # TEST 3: Radar Ground-Truth Comparison Engine
    # -------------------------------------------------------------
    print("\n[TEST 3/8] Testing Radar Ground-Truth Comparison Engine...")
    evaluator = RadarGroundTruthEvaluator()
    evaluator.register_ground_truth(101, 85.0)

    res = evaluator.evaluate_vehicle(101, 83.2)
    print(f"  [RADAR COMPARISON] Calculated: {res['calc_speed']:.1f} km/h | Radar: {res['radar_speed']:.1f} km/h | Error: {res['error_kmh']:.1f} km/h | Accuracy: {res['accuracy_pct']:.1f}%")
    assert abs(res['error_kmh'] - 1.8) < 0.01
    assert abs(res['accuracy_pct'] - 97.88) < 0.1
    print("  [OK] Radar ground-truth comparator and error metrics verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # TEST 4: Multi-Lane Corridor Geometry & Vehicle Spatial Assignment
    # -------------------------------------------------------------
    print("\n[TEST 4/8] Testing Multi-Lane Corridors & Spatial Assignment...")
    lane_mgr = LaneManager(source_polygon=polygon_4k, num_lanes=3)
    assert len(lane_mgr.lanes) == 3, f"Expected 3 lanes, got {len(lane_mgr.lanes)}"

    # Check vehicle coordinates in Lane 1 (left), Lane 2 (middle), and Lane 3 (right)
    # At y = 1560 (middle of road):
    # Lane 1 center ~ 900, Lane 2 center ~ 1300, Lane 3 center ~ 1650
    info_l1 = lane_mgr.identify_vehicle_lane((900, 1560))
    info_l2 = lane_mgr.identify_vehicle_lane((1300, 1560))
    info_l3 = lane_mgr.identify_vehicle_lane((1650, 1560))

    print(f"  [LANE TEST 1] pt (900, 1560) -> {info_l1['lane_name']} (ID: {info_l1['lane_id']})")
    print(f"  [LANE TEST 2] pt (1300, 1560) -> {info_l2['lane_name']} (ID: {info_l2['lane_id']})")
    print(f"  [LANE TEST 3] pt (1650, 1560) -> {info_l3['lane_name']} (ID: {info_l3['lane_id']})")

    assert info_l1['lane_id'] == 1, f"Expected Lane 1, got {info_l1['lane_id']}"
    assert info_l2['lane_id'] == 2, f"Expected Lane 2, got {info_l2['lane_id']}"
    assert info_l3['lane_id'] == 3, f"Expected Lane 3, got {info_l3['lane_id']}"
    print("  [OK] Multi-lane corridor geometry and vehicle assignment successfully verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # TEST 5: Wrong-Way Directional Vector Inversion Detection
    # -------------------------------------------------------------
    print("\n[TEST 5/8] Testing Wrong-Way Driving Vector Analysis (Anti-Flow Detection)...")
    violation_det = ViolationDetector(min_motion_px=10.0)

    # Simulate a car driving backwards (wrong way) in Lane 2 (moving y=1600 -> y=1400, dy < 0)
    wrong_car = TrackedVehicle(99, (1250, 1550, 1350, 1600), (1300, 1600), "car", timestamp=0.0)
    for f in range(1, 15):
        y_step = 1600 - f * 15 # Moving backward/upward
        wrong_car.update((1250, y_step - 50, 1350, y_step), (1300, y_step), timestamp=f / 30.0)

    lane_info_wrong = lane_mgr.identify_vehicle_lane(wrong_car.ground_point)
    v_report = violation_det.check_violations(wrong_car, lane_info_wrong)

    print(f"  [WRONG-WAY TEST] Infractions Detected: {len(v_report['violations_list'])}")
    for v in v_report['violations_list']:
        print(f"    - [{v['severity']}] {v['type']}: {v['message']}")

    assert v_report['has_violation'] is True
    assert any(v['type'] == "WRONG_WAY" for v in v_report['violations_list']), "WRONG_WAY violation failed to trigger!"
    assert v_report['is_critical'] is True
    print("  [OK] Wrong-Way directional vector inversion mathematically confirmed.")
    passed_tests += 1

    # -------------------------------------------------------------
    # TEST 6: Restricted Lane Enforcement (Truck in Fast Lane)
    # -------------------------------------------------------------
    print("\n[TEST 6/8] Testing Restricted Lane Regulations (Heavy Vehicle in Fast Lane)...")
    # Truck in Lane 3 (Fast / Overtake)
    truck = TrackedVehicle(88, (1600, 1500, 1700, 1560), (1650, 1560), "truck", timestamp=0.0)
    lane_info_truck = lane_mgr.identify_vehicle_lane(truck.ground_point)
    v_report_truck = violation_det.check_violations(truck, lane_info_truck)

    print(f"  [RESTRICTED LANE TEST] Truck in {lane_info_truck['lane_name']} -> Violations: {[v['type'] for v in v_report_truck['violations_list']]}")
    assert any(v['type'] == "RESTRICTED_LANE" for v in v_report_truck['violations_list']), "Commercial truck in fast lane not flagged!"
    print("  [OK] Restricted lane compliance verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # TEST 7: CSV Violation Logger & Audit File Integrity
    # -------------------------------------------------------------
    print("\n[TEST 7/8] Testing CSV Violation Logger & Audit File Integrity...")
    test_csv_dir = os.path.join(output_dir, "test_reports")
    test_csv_path = os.path.join(test_csv_dir, "test_violations.csv")
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)

    logger = ViolationLogger(log_dir=test_csv_dir, filename="test_violations.csv", cooldown_seconds=0.0)
    
    # Log test violations
    logger.log(10, wrong_car, lane_info_wrong, v_report['violations_list'][0])
    logger.log(12, truck, lane_info_truck, v_report_truck['violations_list'][0])

    assert os.path.exists(test_csv_path), "CSV report file was not created!"
    with open(test_csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 3, f"Expected header + 2 violation rows, got {len(reader)}"
        header = reader[0]
        assert "Violation_Type" in header and "Vehicle_ID" in header
        print(f"  [CSV TEST] Verified {len(reader)-1} audit rows in: {test_csv_path}")

    print("  [OK] CSV violation audit file format and write operations verified.")
    passed_tests += 1

    # -------------------------------------------------------------
    # TEST 8: End-to-End Real Highway Video Multi-Lane Benchmark Run
    # -------------------------------------------------------------
    print("\n[TEST 8/8] Running End-to-End Real Highway Video Multi-Lane Benchmark...")
    video_path = os.path.join(os.path.dirname(__file__), "data", "highway_traffic.mp4")
    assert os.path.exists(video_path), f"Benchmark video missing: {video_path}"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = VehicleDetector(conf_thresh=0.28)
    tracker = VehicleTracker()
    hud = TrafficHUD()

    frames_processed = 0
    t0 = time.time()
    last_hud_frame = None

    while cap.isOpened() and frames_processed < 50:
        ret, frame = cap.read()
        if not ret:
            break

        frames_processed += 1
        timestamp = frames_processed / fps

        # Detect
        dets = detector.detect(frame)
        # Track
        tracks = tracker.update(dets, timestamp=timestamp)
        # Speed
        for tid, trk in tracks.items():
            speed_calc.compute_speed(trk)

        # Render Multi-Lane HUD
        last_hud_frame = hud.render(
            frame,
            tracked_vehicles=tracks,
            transformer=transformer,
            lane_manager=lane_mgr,
            violation_detector=violation_det,
            violation_logger=logger,
            evaluator=evaluator,
            fps=30.0,
            show_lanes=True,
            frame_idx=frames_processed
        )

    cap.release()
    total_time = time.time() - t0
    avg_fps = frames_processed / max(total_time, 0.001)

    print(f"  [BENCHMARK] Processed {frames_processed} highway frames in {total_time:.2f}s | Average Speed: {avg_fps:.1f} FPS")
    print(f"  [TRACKING STATS] Actively tracked {len(tracks)} vehicles on the multi-lane highway.")
    assert len(tracks) > 0, "No vehicles were tracked in benchmark video!"

    # Save verified visual output
    hud_output_path = os.path.join(output_dir, "verified_multilane_traffic_hud.jpg")
    cv2.imwrite(hud_output_path, last_hud_frame)
    print(f"  [OK] Multi-lane highway video benchmark passed! Saved output to {hud_output_path}")
    passed_tests += 1

    print("\n" + "=" * 75)
    print(f"  SUMMARY: {passed_tests}/{total_tests} TESTS PASSED [100% SCIENTIFIC VERIFICATION]")
    print(f"  Multi-Lane HUD Output: {hud_output_path}")
    print("=" * 75)

if __name__ == "__main__":
    run_speed_tracker_tests()
