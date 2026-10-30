import numpy as np

class ViolationDetector:
    """
    Evaluates vehicle behavior against traffic laws:
    - Wrong-Way Driving (anti-flow motion vector dot product)
    - Over-speeding (lane-specific speed thresholds)
    - Restricted Lane Access (commercial trucks / buses in overtaking lane)
    - Lane Straddling (driving on white dividing markers)
    """
    def __init__(self, min_motion_px=10.0, wrong_way_cos_threshold=-0.25):
        """
        Args:
            min_motion_px: Minimum trajectory displacement required to evaluate direction
            wrong_way_cos_threshold: cos(theta) threshold below which motion is opposite flow
        """
        self.min_motion_px = float(min_motion_px)
        self.wrong_way_threshold = float(wrong_way_cos_threshold)

    def check_violations(self, vehicle, lane_info):
        """
        Checks a tracked vehicle for traffic infractions.
        
        Args:
            vehicle: TrackedVehicle instance
            lane_info: dict returned by LaneManager.identify_vehicle_lane()
            
        Returns:
            dict containing:
                - 'has_violation': bool
                - 'is_critical': bool (True for WRONG_WAY or severe speeding)
                - 'primary_violation': str or None
                - 'violations_list': list of dicts with details
        """
        violations = []
        lane = lane_info.get("lane")
        speed = vehicle.current_speed_kmh

        # 1. WRONG-WAY DRIVING DETECTION (Vector Heading Analysis)
        if lane is not None and len(vehicle.history) >= 5:
            # Take displacement between oldest available and newest frame
            hist_start = vehicle.history[0]
            hist_end = vehicle.history[-1]
            
            dx = float(hist_end[0] - hist_start[0])
            dy = float(hist_end[1] - hist_start[1])
            disp = np.sqrt(dx * dx + dy * dy)

            if disp >= self.min_motion_px:
                # Dot product between motion unit vector and lane flow unit vector
                unit_motion = np.array([dx, dy]) / disp
                cos_theta = float(np.dot(unit_motion, lane.flow_unit))
                
                # If cos(theta) < -0.25 (angle > 105 degrees from traffic flow)
                if cos_theta < self.wrong_way_threshold:
                    violations.append({
                        "type": "WRONG_WAY",
                        "severity": "CRITICAL",
                        "code": "WW_01",
                        "message": f"WRONG WAY! Vehicle moving against traffic flow (angle: {np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))):.0f}°)"
                    })

        # 2. OVER-SPEEDING DETECTION
        if speed is not None and lane is not None:
            speed_limit = lane.speed_limit_kmh
            if speed > speed_limit:
                excess = speed - speed_limit
                severity = "CRITICAL" if excess > 25.0 else ("HIGH" if excess > 10.0 else "MEDIUM")
                violations.append({
                    "type": "OVER_SPEEDING",
                    "severity": severity,
                    "code": "SPD_01",
                    "message": f"Over-Speeding: {speed:.1f} km/h (Limit: {speed_limit:.0f} km/h, +{excess:.1f} km/h)"
                })

        # 3. RESTRICTED LANE ACCESS
        if lane is not None and vehicle.class_name.lower() in lane.restricted_classes:
            violations.append({
                "type": "RESTRICTED_LANE",
                "severity": "HIGH",
                "code": "LAN_01",
                "message": f"Illegal Lane: {vehicle.class_name.upper()} prohibited in {lane.name}"
            })

        # 4. LANE STRADDLING
        if lane_info.get("is_straddling", False):
            violations.append({
                "type": "LANE_STRADDLING",
                "severity": "LOW",
                "code": "LAN_02",
                "message": f"Lane Straddling: Vehicle driving over divider line"
            })

        has_violation = len(violations) > 0
        is_critical = any(v["severity"] == "CRITICAL" for v in violations)
        primary_violation = violations[0]["type"] if has_violation else None

        return {
            "has_violation": has_violation,
            "is_critical": is_critical,
            "primary_violation": primary_violation,
            "violations_list": violations
        }
