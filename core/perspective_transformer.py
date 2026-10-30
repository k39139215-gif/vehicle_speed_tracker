import cv2
import numpy as np

class PerspectiveTransformer:
    """
    Inverse Perspective Mapping (IPM) & Road Homography Calibrator.
    Maps 2D camera pixels (x, y) into physical real-world Euclidean coordinates (X, Y)
    in meters, compensating for perspective foreshortening and camera tilt.
    """
    def __init__(self, source_polygon=None, real_length_meters=80.0, real_width_meters=14.0):
        """
        Args:
            source_polygon: 4 (x, y) coordinates of the road trapezoid in the camera view:
                            [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
            real_length_meters: Physical length of the road section in meters
            real_width_meters: Physical width of the road section in meters
        """
        # Default calibrated polygon covering the incoming traffic lanes in 4K UHD (3840 x 2160)
        if source_polygon is None:
            self.source_polygon = np.array([
                [1350, 960],    # Top-Left (Left lane under gantry)
                [1780, 960],    # Top-Right (Divider under gantry)
                [1950, 2160],   # Bottom-Right (Divider near camera)
                [100, 2160]     # Bottom-Left (Left roadside near camera)
            ], dtype=np.float32)
        else:
            self.source_polygon = np.array(source_polygon, dtype=np.float32)

        self.real_length = float(real_length_meters)
        self.real_width = float(real_width_meters)

        # Destination rectangle in real-world metric space [0, real_width] x [0, real_length]
        self.dest_polygon = np.array([
            [0.0, 0.0],
            [self.real_width, 0.0],
            [self.real_width, self.real_length],
            [0.0, self.real_length]
        ], dtype=np.float32)

        # Compute Homography Matrix H
        self.H = cv2.getPerspectiveTransform(self.source_polygon, self.dest_polygon)

    def transform_point(self, px, py):
        """
        Transforms a camera pixel coordinate (px, py) to real-world metric coordinates (X, Y)
        in meters using the camera's calibrated homography matrix H.
        
        Returns:
            (real_x, real_y): Coordinates in meters along road width and road length.
        """
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        real_x = float(transformed[0, 0, 0])
        real_y = float(transformed[0, 0, 1])
        return real_x, real_y

    def get_road_depth(self, px, py, is_approaching=None):
        """
        Returns metric longitudinal position Y along the calibrated road plane in meters.
        Universal across all camera angles, resolutions, and perspectives.
        """
        _, real_y = self.transform_point(px, py)
        return real_y

    def calculate_road_distance(self, pt1_px, pt2_px):
        """
        Calculates real-world Euclidean distance in meters between two camera pixel points
        on the calibrated ground plane.
        """
        x1, y1 = self.transform_point(pt1_px[0], pt1_px[1])
        x2, y2 = self.transform_point(pt2_px[0], pt2_px[1])
        dx = x2 - x1
        dy = y2 - y1
        return float(np.sqrt(dx * dx + dy * dy))

    def calculate_real_distance(self, pt1_px, pt2_px):
        """
        Calculates the real Euclidean distance in meters between two camera pixel points.
        """
        return self.calculate_road_distance(pt1_px, pt2_px)

    def is_inside_calibration_zone(self, px, py):
        """
        Checks whether a point lies inside the calibrated road polygon.
        """
        pt = (float(px), float(py))
        res = cv2.pointPolygonTest(self.source_polygon.astype(np.int32), pt, False)
        return res >= 0

    def draw_calibration_zone(self, frame_bgr, color=(0, 220, 255), thickness=2):
        """
        Draws the active calibration trapezoid and speed detection zone on the frame.
        """
        pts = self.source_polygon.astype(np.int32).reshape((-1, 1, 2))
        
        # Semi-transparent overlay inside zone
        overlay = frame_bgr.copy()
        cv2.fillPoly(overlay, [pts], (20, 45, 10))
        cv2.addWeighted(overlay, 0.25, frame_bgr, 0.75, 0, frame_bgr)

        # Draw boundary lines
        cv2.polylines(frame_bgr, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        # Draw Corner Markers & Distance Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_bgr, "SPEED MEASUREMENT ZONE (80m)", (int(self.source_polygon[0][0]) - 50, int(self.source_polygon[0][1]) - 12),
                    font, 0.55, color, 2, cv2.LINE_AA)
        return frame_bgr
