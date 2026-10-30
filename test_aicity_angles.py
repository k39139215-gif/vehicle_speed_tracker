import cv2
import os
import sys
import numpy as np

# Add project root
tracker_dir = os.path.dirname(__file__)
if tracker_dir not in sys.path:
    sys.path.append(tracker_dir)

from core.vehicle_detector import VehicleDetector
from core.vehicle_tracker import VehicleTracker

def test_aicity_extreme_angles():
    print("=" * 80)
    print("  STRESS-TESTING YOLOv8 + TRACKING ON 5 EXTREME AI CITY CHALLENGE ANGLES")
    print("=" * 80)

    detector = VehicleDetector(conf_thresh=0.22)
    tracker = VehicleTracker(max_distance_threshold=65.0, max_lost_frames=12)

    data_dir = os.path.join(tracker_dir, "data", "aicity_challenge")
    results_dir = os.path.join(tracker_dir, "reports", "aicity_stress_tests")
    os.makedirs(results_dir, exist_ok=True)

    videos = [
        ("angle1_distant_highway_cctv.mp4", "Angle 1: Distant Freeway CCTV (320x240 Low-Res)"),
        ("angle2_dawn_lowlight_intersection.mp4", "Angle 2: Pitch-Black Nocturnal CCTV (02:38 AM Glare)"),
        ("angle3_congested_roundabout.mp4", "Angle 3: Complex Multi-Lane Roundabout (Curved Trajectories)"),
        ("angle4_acute_oblique_anomaly.mp4", "Angle 4: Oblique Pole View (Trucks + Scooters + Cars)"),
        ("angle5_close_quarters_occlusion.mp4", "Angle 5: Multi-Lane Arterial (Pedestrians + Mixed Traffic)")
    ]

    summary = []

    for idx, (vfile, desc) in enumerate(videos, 1):
        vpath = os.path.join(data_dir, vfile)
        if not os.path.exists(vpath):
            print(f"[SKIP] {vfile} not found")
            continue

        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n[{idx}/5] Evaluating {desc}")
        print(f"      Source: {vfile} ({w}x{h} @ {fps:.1f} FPS, {total_frames} frames)")

        # Process first 90 frames
        f_idx = 0
        all_tracked_ids = set()
        total_detections = 0
        annotated_frame = None

        while cap.isOpened() and f_idx < min(90, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            f_idx += 1
            timestamp = f_idx / fps

            dets = detector.detect(frame)
            total_detections += len(dets)
            active_tracks = tracker.update(dets, timestamp=timestamp)

            for tid in active_tracks.keys():
                all_tracked_ids.add(tid)

            # Annotate mid frame for visual proof
            if f_idx == min(45, total_frames // 2):
                annotated_frame = frame.copy()
                for tid, trk in active_tracks.items():
                    bx1, by1, bx2, by2 = trk.bbox
                    color = (0, 255, 120) if trk.class_name == "car" else ((0, 140, 255) if trk.class_name == "truck" else (0, 220, 255))
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
                    label = f"ID #{tid} | {trk.class_name.upper()}"
                    cv2.putText(annotated_frame, label, (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                
                # Header tag
                cv2.putText(annotated_frame, f"AI CITY CHALLENGE: {desc}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 2)

        cap.release()

        # Save annotated image
        out_img_path = os.path.join(results_dir, f"{os.path.splitext(vfile)[0]}_tracked.jpg")
        if annotated_frame is not None:
            cv2.imwrite(out_img_path, annotated_frame)

        avg_dets_per_frame = total_detections / max(1, f_idx)
        print(f"      Results: Tracked {len(all_tracked_ids)} unique vehicles across {f_idx} frames (Avg {avg_dets_per_frame:.1f} cars/frame)")
        print(f"      Annotated Visual Saved: {out_img_path}")

        summary.append({
            "name": desc,
            "filename": vfile,
            "resolution": f"{w}x{h}",
            "vehicles_tracked": len(all_tracked_ids),
            "avg_detections_frame": round(avg_dets_per_frame, 1),
            "output_image": out_img_path
        })

    print("\n" + "=" * 80)
    print("  STRESS TEST SUMMARY ACROSS 5 EXTREME ANGLES:")
    print("=" * 80)
    for s in summary:
        print(f"  * {s['name']:60s} | {s['resolution']:9s} | {s['vehicles_tracked']} vehicles tracked (Avg: {s['avg_detections_frame']}/frame)")
    print("=" * 80)

if __name__ == "__main__":
    test_aicity_extreme_angles()
