import cv2
import numpy as np

class TrafficHUD:
    """
    Renders high-contrast AI vehicle speed tracking and traffic enforcement HUD:
    - Clean road view (ZERO distracting lane fills or polygon masks)
    - High-visibility bounding boxes with corner brackets
    - Large, bold, high-contrast speed badges (XX km/h) readable at any distance
    - Prominent overspeeding alerts and real-time FPS throughput
    """
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._flash_counter = 0

    def render(self, frame_bgr, tracked_vehicles, transformer, lane_manager,
               violation_detector, violation_logger, evaluator,
               fps=30.0, show_lanes=False, frame_idx=0, speed_limit_kmh=80.0):
        h, w = frame_bgr.shape[:2]
        self._flash_counter = (self._flash_counter + 1) % 30
        is_flash_on = (self._flash_counter < 15)

        # 1. Subtle Lane Lines ONLY if user explicitly turns them on (Zero filled polygons!)
        if show_lanes and lane_manager is not None:
            for idx, lane in enumerate(lane_manager.lanes):
                cv2.polylines(frame_bgr, [lane.polygon], isClosed=True, color=(0, 200, 255), thickness=1, lineType=cv2.LINE_AA)

        # 2. Draw Vehicle Bounding Boxes, Corner Brackets, and BOLD Speed Badges
        for tid, trk in tracked_vehicles.items():
            x1, y1, x2, y2 = trk.bbox
            speed = trk.current_speed_kmh

            # Identify lane and infractions (silent logging without road tinting)
            lane_info = lane_manager.identify_vehicle_lane(trk.ground_point) if lane_manager else {"lane_id": 0, "lane_name": "N/A", "lane": None}
            v_report = violation_detector.check_violations(trk, lane_info) if violation_detector else {"has_violation": False, "violations_list": []}

            if violation_logger and v_report["has_violation"]:
                for v in v_report["violations_list"]:
                    violation_logger.log(frame_idx, trk, lane_info, v)

            # Determine overspeeding status
            is_overspeeding = (speed is not None and speed > speed_limit_kmh)

            # Setup clean, prominent badge colors matching YouTube CV demos
            if is_overspeeding:
                box_color = (0, 40, 245)       # Vivid Crimson Red
                bg_color = (0, 10, 180)        # Solid Crimson badge
                text_color = (255, 255, 255)   # Bold White text
                tag_text = f"#{tid:02d} | {speed:.0f} km/h (FAST)"
            elif speed is not None and speed >= 5.0:
                box_color = (0, 240, 120)      # Bright Neon Green
                bg_color = (15, 22, 28)        # Sleek Dark Pill
                text_color = (0, 255, 180)     # Crisp Neon Green
                tag_text = f"#{tid:02d} | {speed:.0f} km/h"
            else:
                # Active tracking badge
                box_color = (0, 220, 255)      # Cyan
                bg_color = (15, 22, 28)
                text_color = (0, 230, 255)
                tag_text = f"#{tid:02d} | {trk.class_name.upper()}"

            # Sleek bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # High-tech corner brackets
            c_len = min(16, max(6, (x2 - x1) // 3))
            c_thick = 3
            cv2.line(frame_bgr, (x1, y1), (x1 + c_len, y1), box_color, c_thick)
            cv2.line(frame_bgr, (x1, y1), (x1, y1 + c_len), box_color, c_thick)
            cv2.line(frame_bgr, (x2, y1), (x2 - c_len, y1), box_color, c_thick)
            cv2.line(frame_bgr, (x2, y1), (x2, y1 + c_len), box_color, c_thick)
            cv2.line(frame_bgr, (x1, y2), (x1 + c_len, y2), box_color, c_thick)
            cv2.line(frame_bgr, (x1, y2), (x1, y2 - c_len), box_color, c_thick)
            cv2.line(frame_bgr, (x2, y2), (x2 - c_len, y2), box_color, c_thick)
            cv2.line(frame_bgr, (x2, y2), (x2, y2 - c_len), box_color, c_thick)

            # Ground contact tire dot
            cv2.circle(frame_bgr, trk.ground_point, 4, box_color, -1, cv2.LINE_AA)

            # Render BIG, BOLD SPEED BADGE (Enlarged & Screen-Clamped)
            font_scale = 0.65
            font_thick = 2
            (tw, th), baseline = cv2.getTextSize(tag_text, self.font, font_scale, font_thick)

            # Prevent badge from getting cut off at screen borders
            badge_x = min(max(8, x1), w - tw - 16)
            badge_y = max(th + 16, y1 - 10)

            # Badge pill background & border
            cv2.rectangle(frame_bgr, (badge_x - 5, badge_y - th - 6), (badge_x + tw + 6, badge_y + baseline + 2), bg_color, -1)
            cv2.rectangle(frame_bgr, (badge_x - 5, badge_y - th - 6), (badge_x + tw + 6, badge_y + baseline + 2), box_color, 2, cv2.LINE_AA)

            # High-contrast bold text inside badge
            cv2.putText(frame_bgr, tag_text, (badge_x, badge_y - 2), self.font, font_scale, text_color, font_thick, cv2.LINE_AA)

        # 3. Sleek Top Glassmorphic Telemetry Header Bar
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (12, 16, 22), -1)
        cv2.addWeighted(overlay, 0.78, frame_bgr, 0.22, 0, frame_bgr)
        cv2.line(frame_bgr, (0, 50), (w, 50), (0, 230, 255), 1, cv2.LINE_AA)

        # Header titles and metrics
        cv2.putText(frame_bgr, "AI SPEED RADAR & VEHICLE TRACKER", (20, 32), self.font, 0.68, (0, 240, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"SPEED LIMIT: {int(speed_limit_kmh)} KM/H  |  TRACKED: {len(tracked_vehicles)} VEHICLES",
                    (max(400, w // 2 - 140), 32), self.font, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        
        # FPS Indicator
        fps_color = (0, 255, 150) if fps >= 24.0 else (0, 200, 255)
        cv2.putText(frame_bgr, f"FPS: {fps:04.1f}", (w - 140, 32), self.font, 0.62, fps_color, 2, cv2.LINE_AA)

        # 4. Bottom Clean Controls Bar
        cv2.rectangle(frame_bgr, (0, h - 30), (w, h), (12, 16, 22), -1)
        controls_msg = "[N] NEXT CAM  |  [P] PREV CAM  |  [1-9] JUMP  |  [SPACE] PAUSE  |  [L] TOGGLE LANES  |  [Q] EXIT"
        cv2.putText(frame_bgr, controls_msg, (20, h - 10), self.font, 0.44, (180, 220, 240), 1, cv2.LINE_AA)

        return frame_bgr
