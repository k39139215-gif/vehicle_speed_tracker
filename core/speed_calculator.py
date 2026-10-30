import numpy as np

class SpeedCalculator:
    """
    Universal State-of-the-Art ITS Vehicle Speed Estimator:
    1. Inverse-Perspective Mapping (IPM) transforms pixel coordinates into real-world 
       Euclidean ground coordinates (X, Y) in meters, compensating for perspective foreshortening.
    2. 2D Moving-Window Ordinary Least Squares (OLS) Linear Regression:
       X(t) = v_x * t + X_0
       Y(t) = v_y * t + Y_0
       Speed magnitude v = sqrt(v_x^2 + v_y^2) * 3.6 km/h.
       Works universally across straight highways, roundabouts, curved ramps, and diagonal intersections.
    3. Metric Convoy / Platoon Kinematic Coupling (TeamUW AI City Challenge & Roboflow principle):
       In metric space, vehicles in the same lane (lateral offset < 3.5m) traveling in the same direction
       with gap < 30m cannot exceed the lead vehicle's speed plus closure rate. This guarantees that a
       trailing vehicle will NEVER report a higher speed (e.g. 175 km/h behind a 65 km/h truck).
    4. Physical Inertia Rate-Limiter (Kinematic Acceleration Clamp):
       Acceleration is clamped to physical vehicle bounds (|a| <= max_accel, default 7.2 km/h/s = 2.0 m/s^2)
       so speeds never spike or plummet abruptly from detector jitter.
    5. Road Boundary Freezing:
       Maintains last valid speed when a vehicle reaches the boundary of the calibration zone
       to prevent clipped bounding box deceleration.
    """
    def __init__(self, perspective_transformer, fps=30.0, speed_limit_kmh=80.0,
                 min_history_frames=8, window_size=12, max_accel_kmh_s=7.2):
        self.transformer = perspective_transformer
        self.fps = float(fps)
        self.speed_limit = float(speed_limit_kmh)
        self.min_history = max(int(min_history_frames), 6)
        self.window_size = int(window_size)
        self.max_accel_kmh_s = float(max_accel_kmh_s)

    def compute_speed(self, tracked_vehicle, all_active_tracks=None):
        """
        Calculates physically realistic, smoothed speed in km/h for a tracked vehicle.
        
        Returns:
            speed_kmh (float or None): Current speed in km/h, or None if insufficient history.
            is_overspeeding (bool): True if speed exceeds the legal speed limit.
        """
        history = tracked_vehicle.history
        if len(history) < self.min_history:
            return None, False

        curr_px, curr_py, t_curr = history[-1]

        # Check if vehicle has reached or exited the calibration boundary
        # If so, maintain its last calculated valid speed
        if hasattr(self.transformer, 'is_inside_calibration_zone'):
            if not self.transformer.is_inside_calibration_zone(curr_px, curr_py):
                if tracked_vehicle.current_speed_kmh is not None:
                    return tracked_vehicle.current_speed_kmh, tracked_vehicle.current_speed_kmh > self.speed_limit

        # Take moving window of recent track points
        lookback = min(len(history), self.window_size)
        pts = history[-lookback:]

        times = np.array([p[2] for p in pts], dtype=np.float64)
        
        # Transform all window points into real-world metric coordinates (X, Y) in meters
        metric_pts = [self.transformer.transform_point(p[0], p[1]) for p in pts]
        xs = np.array([p[0] for p in metric_pts], dtype=np.float64)
        ys = np.array([p[1] for p in metric_pts], dtype=np.float64)

        # Robust 2D OLS linear regression:
        # X(t) = v_x * t + c_x, Y(t) = v_y * t + c_y
        dt_arr = times - np.mean(times)
        var_t = np.sum(dt_arr ** 2)

        if var_t < 1e-5:
            return tracked_vehicle.current_speed_kmh, False

        vx_mps = np.sum(dt_arr * (xs - np.mean(xs))) / var_t
        vy_mps = np.sum(dt_arr * (ys - np.mean(ys))) / var_t

        v_mag_mps = float(np.sqrt(vx_mps ** 2 + vy_mps ** 2))
        raw_speed_kmh = v_mag_mps * 3.6

        # Physical sanity bounds: Ignore non-physical outliers (> 200 km/h) or sub-pixel noise (< 3 km/h)
        if raw_speed_kmh > 200.0:
            return tracked_vehicle.current_speed_kmh, False
        if raw_speed_kmh < 3.0:
            raw_speed_kmh = 0.0

        # Convoy / Platoon Kinematic Coupling Constraint:
        # If following a lead vehicle in the same lane corridor within 30m headway,
        # the trailing car's speed is physically bounded by the lead car's speed + closure rate.
        if all_active_tracks is not None and v_mag_mps > 1.0:
            curr_x, curr_y = metric_pts[-1]
            u_x = vx_mps / v_mag_mps
            u_y = vy_mps / v_mag_mps

            lead_vehicle = None
            min_headway = 999.0

            for other_tid, other_trk in all_active_tracks.items():
                if other_tid == tracked_vehicle.track_id or other_trk.current_speed_kmh is None:
                    continue
                if len(other_trk.history) < 2:
                    continue
                ox, oy, _ = other_trk.history[-1]
                other_mx, other_my = self.transformer.transform_point(ox, oy)

                dx = other_mx - curr_x
                dy = other_my - curr_y

                # Longitudinal projection s along travel direction
                s_longitudinal = dx * u_x + dy * u_y
                # Lateral perpendicular offset (lane deviation)
                d_lateral = abs(-dx * u_y + dy * u_x)

                # Same lane (lateral offset < 3.5m) and ahead along motion (1m < s < 30m)
                if d_lateral < 3.5 and 1.0 < s_longitudinal < 30.0:
                    if s_longitudinal < min_headway:
                        min_headway = s_longitudinal
                        lead_vehicle = other_trk

            if lead_vehicle is not None and lead_vehicle.current_speed_kmh is not None:
                v_lead = lead_vehicle.current_speed_kmh
                if not hasattr(tracked_vehicle, 'gap_history'):
                    tracked_vehicle.gap_history = []
                tracked_vehicle.gap_history.append((t_curr, min_headway))
                if len(tracked_vehicle.gap_history) > 15:
                    tracked_vehicle.gap_history.pop(0)

                # Compute rate of gap change d(gap)/dt via regression
                if len(tracked_vehicle.gap_history) >= 6:
                    g_pts = tracked_vehicle.gap_history[-6:]
                    g_times = np.array([p[0] for p in g_pts])
                    g_vals = np.array([p[1] for p in g_pts])
                    d_g_times = g_times - np.mean(g_times)
                    denom_g = np.sum(d_g_times ** 2)
                    d_gap_dt = (np.sum(d_g_times * (g_vals - np.mean(g_vals))) / denom_g) if denom_g > 1e-5 else 0.0
                    rel_speed_kmh = -d_gap_dt * 3.6  # Negative rate of gap expansion = closure speed
                    expected_speed = v_lead + rel_speed_kmh
                    # Clamp raw speed tightly to physical convoy expectation (+/- 4 km/h)
                    raw_speed_kmh = np.clip(raw_speed_kmh, expected_speed - 4.0, expected_speed + 4.0)
                else:
                    # Initial grace clamp: trailing vehicle cannot exceed lead vehicle by more than 4 km/h
                    raw_speed_kmh = np.clip(raw_speed_kmh, v_lead - 10.0, v_lead + 4.0)

        # First speed initialization
        if tracked_vehicle.current_speed_kmh is None:
            tracked_vehicle.current_speed_kmh = raw_speed_kmh
            tracked_vehicle.speeds_recorded.append(raw_speed_kmh)
            tracked_vehicle.average_speed_kmh = raw_speed_kmh
            tracked_vehicle.last_speed_time = t_curr
            return raw_speed_kmh, raw_speed_kmh > self.speed_limit

        # Physical Inertia Rate-Limiter (Kinematic Acceleration Clamp)
        # Vehicles cannot physically accelerate or brake faster than a_max (e.g. 7.2 km/h per second)
        last_t = getattr(tracked_vehicle, 'last_speed_time', None)
        dt = (t_curr - last_t) if last_t is not None else (1.0 / self.fps)
        if dt <= 0.001:
            dt = 1.0 / self.fps
        tracked_vehicle.last_speed_time = t_curr

        max_allowed_change = self.max_accel_kmh_s * dt
        speed_delta = raw_speed_kmh - tracked_vehicle.current_speed_kmh
        speed_delta_clamped = np.clip(speed_delta, -max_allowed_change, max_allowed_change)

        smoothed_speed = tracked_vehicle.current_speed_kmh + speed_delta_clamped

        tracked_vehicle.current_speed_kmh = smoothed_speed
        tracked_vehicle.speeds_recorded.append(smoothed_speed)
        tracked_vehicle.average_speed_kmh = float(np.mean(tracked_vehicle.speeds_recorded))

        is_overspeeding = smoothed_speed > self.speed_limit
        return smoothed_speed, is_overspeeding
