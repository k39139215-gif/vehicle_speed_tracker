import cv2
import numpy as np

class Lane:
    """
    Represents an individual structured highway lane corridor.
    """
    def __init__(self, lane_id, name, polygon, flow_vector, speed_limit_kmh, restricted_classes=None):
        self.lane_id = lane_id
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self.flow_vector = np.array(flow_vector, dtype=np.float32)
        norm = np.linalg.norm(self.flow_vector)
        self.flow_unit = self.flow_vector / (norm + 1e-6)
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.restricted_classes = set(restricted_classes or [])

class LaneManager:
    """
    Manages multi-lane highway layout, vehicle-to-lane assignment,
    divider straddling detection, and directional heading compliance.
    """
    def __init__(self, source_polygon=None, num_lanes=3):
        """
        Args:
            source_polygon: 4 road corners [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
            num_lanes: Number of structured longitudinal lanes (default 3)
        """
        if source_polygon is None:
            self.road_polygon = np.array([
                [1350, 960],    # Top-Left
                [1780, 960],    # Top-Right
                [1950, 2160],   # Bottom-Right
                [100, 2160]     # Bottom-Left
            ], dtype=np.float32)
        else:
            self.road_polygon = np.array(source_polygon, dtype=np.float32)

        self.num_lanes = num_lanes
        self.lanes = []
        self._build_lanes()

    def _build_lanes(self):
        """
        Subdivides the 4-point road trapezoid into N structured longitudinal corridors.
        """
        self.lanes.clear()
        
        # Top edge: p0 (Top-Left) to p1 (Top-Right)
        p_top_l = self.road_polygon[0]
        p_top_r = self.road_polygon[1]
        
        # Bottom edge: p3 (Bottom-Left) to p2 (Bottom-Right)
        p_bot_r = self.road_polygon[2]
        p_bot_l = self.road_polygon[3]

        lane_configs = [
            {"name": "Lane 1 (Slow / Trucks)", "speed_limit": 70.0, "restricted": []},
            {"name": "Lane 2 (Cruising)", "speed_limit": 80.0, "restricted": ["truck"]},
            {"name": "Lane 3 (Fast / Overtake)", "speed_limit": 90.0, "restricted": ["truck", "bus"]}
        ]

        for i in range(self.num_lanes):
            # Interpolate top edge
            alpha_0 = i / float(self.num_lanes)
            alpha_1 = (i + 1) / float(self.num_lanes)
            
            top_start = (1.0 - alpha_0) * p_top_l + alpha_0 * p_top_r
            top_end   = (1.0 - alpha_1) * p_top_l + alpha_1 * p_top_r
            
            # Interpolate bottom edge
            bot_start = (1.0 - alpha_0) * p_bot_l + alpha_0 * p_bot_r
            bot_end   = (1.0 - alpha_1) * p_bot_l + alpha_1 * p_bot_r

            corridor = [
                [int(top_start[0]), int(top_start[1])],
                [int(top_end[0]),   int(top_end[1])],
                [int(bot_end[0]),   int(bot_end[1])],
                [int(bot_start[0]), int(bot_start[1])]
            ]

            # Flow vector is from top center to bottom center
            center_top = (top_start + top_end) / 2.0
            center_bot = (bot_start + bot_end) / 2.0
            flow_vec = center_bot - center_top

            config = lane_configs[min(i, len(lane_configs) - 1)]
            lane = Lane(
                lane_id=i + 1,
                name=config["name"],
                polygon=corridor,
                flow_vector=flow_vec,
                speed_limit_kmh=config["speed_limit"],
                restricted_classes=config["restricted"]
            )
            self.lanes.append(lane)

    def reconfigure(self, new_polygon):
        """
        Updates lane corridors when the user recalibrates the road with mouse clicks.
        """
        self.road_polygon = np.array(new_polygon, dtype=np.float32)
        self._build_lanes()

    def identify_vehicle_lane(self, point_px, straddle_margin_px=22.0):
        """
        Determines which lane a vehicle's tire-road contact point occupies.
        
        Args:
            point_px: (x, y) coordinates of the bottom-center of the vehicle bounding box.
            straddle_margin_px: Pixel distance from boundary to flag lane straddling.
            
        Returns:
            dict with:
                - 'lane': Lane object or None
                - 'lane_id': int (1, 2, 3...) or 0 if outside calibrated road
                - 'lane_name': str
                - 'is_straddling': bool
                - 'distance_to_edge': float
        """
        px, py = float(point_px[0]), float(point_px[1])
        
        best_lane = None
        best_dist = -9999.0
        
        for lane in self.lanes:
            dist = cv2.pointPolygonTest(lane.polygon, (px, py), measureDist=True)
            if dist >= 0:
                # Inside this lane polygon
                best_lane = lane
                best_dist = dist
                break
            elif dist > best_dist:
                # Closest lane if slightly outside
                best_dist = dist
                best_lane = lane

        if best_lane is None or best_dist < -40.0:
            return {
                "lane": None,
                "lane_id": 0,
                "lane_name": "Off-Road / Shoulder",
                "is_straddling": False,
                "distance_to_edge": best_dist
            }

        # Check if vehicle is straddling the dividing line
        is_straddling = (best_dist >= 0 and best_dist < straddle_margin_px)

        return {
            "lane": best_lane,
            "lane_id": best_lane.lane_id,
            "lane_name": best_lane.name,
            "is_straddling": is_straddling,
            "distance_to_edge": best_dist
        }
