import numpy as np
import time

class TrackedVehicle:
    """
    Represents a single persistent vehicle track.
    """
    def __init__(self, track_id, bbox, ground_point, class_name, timestamp):
        self.track_id = track_id
        self.bbox = bbox
        self.ground_point = ground_point
        self.class_name = class_name
        
        # Trajectory history: list of tuples (ground_x, ground_y, timestamp)
        self.history = [(ground_point[0], ground_point[1], timestamp)]
        self.lost_frames = 0
        self.current_speed_kmh = None
        self.average_speed_kmh = None
        self.speeds_recorded = []

    def update(self, bbox, ground_point, timestamp):
        self.bbox = bbox
        self.ground_point = ground_point
        self.history.append((ground_point[0], ground_point[1], timestamp))
        # Keep recent history for trail rendering (last 30 points)
        if len(self.history) > 45:
            self.history.pop(0)
        self.lost_frames = 0

class VehicleTracker:
    """
    Robust centroid and proximity-based multi-object vehicle tracker.
    Maintains persistent vehicle identities (IDs) and trajectories across video frames.
    """
    def __init__(self, max_distance_threshold=65.0, max_lost_frames=12):
        self.max_distance = max_distance_threshold
        self.max_lost = max_lost_frames
        self.tracks = {}
        self.next_id = 1

    def update(self, detected_vehicles, timestamp=None):
        """
        Matches detected vehicles to existing tracks or creates new ones.
        
        Args:
            detected_vehicles: list of dicts from VehicleDetector
            timestamp: frame timestamp in seconds (float)
            
        Returns:
            dict of active TrackedVehicle objects keyed by track_id
        """
        if timestamp is None:
            timestamp = time.time()

        if len(detected_vehicles) == 0:
            # Increment lost counter for all existing tracks
            lost_ids = []
            for tid, trk in self.tracks.items():
                trk.lost_frames += 1
                if trk.lost_frames > self.max_lost:
                    lost_ids.append(tid)
            for tid in lost_ids:
                del self.tracks[tid]
            return self.tracks

        # If no active tracks, initialize all detections as new tracks
        if len(self.tracks) == 0:
            for det in detected_vehicles:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = TrackedVehicle(
                    tid, det['bbox'], det['ground_point'], det['class_name'], timestamp
                )
            return self.tracks

        # Match existing tracks with detections via Euclidean distance of ground points
        track_ids = list(self.tracks.keys())
        track_points = np.array([self.tracks[tid].ground_point for tid in track_ids])
        det_points = np.array([det['ground_point'] for det in detected_vehicles])

        # Distance matrix (N_tracks x M_detections)
        dist_matrix = np.linalg.norm(track_points[:, np.newaxis] - det_points[np.newaxis, :], axis=2)

        matched_tracks = set()
        matched_dets = set()

        # Greedy matching by minimum distance
        while True:
            if dist_matrix.size == 0:
                break
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            trk_idx, det_idx = min_idx
            min_dist = dist_matrix[trk_idx, det_idx]

            if min_dist > self.max_distance or min_dist == np.inf:
                break

            tid = track_ids[trk_idx]
            det = detected_vehicles[det_idx]

            self.tracks[tid].update(det['bbox'], det['ground_point'], timestamp)
            matched_tracks.add(trk_idx)
            matched_dets.add(det_idx)

            # Invalidate row and column
            dist_matrix[trk_idx, :] = np.inf
            dist_matrix[:, det_idx] = np.inf

        # Handle unmatched detections (new vehicles entered scene)
        for det_idx, det in enumerate(detected_vehicles):
            if det_idx not in matched_dets:
                # Check for overlap with existing tracks to prevent duplicate nested boxes
                nx1, ny1, nx2, ny2 = det['bbox']
                n_area = (nx2 - nx1) * (ny2 - ny1)
                is_duplicate = False
                for ex_tid, ex_trk in self.tracks.items():
                    ex1, ey1, ex2, ey2 = ex_trk.bbox
                    ix1 = max(nx1, ex1)
                    iy1 = max(ny1, ey1)
                    ix2 = min(nx2, ex2)
                    iy2 = min(ny2, ey2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if inter_area / max(1.0, min(n_area, (ex2 - ex1) * (ey2 - ey1))) > 0.55:
                            is_duplicate = True
                            break
                if is_duplicate:
                    continue

                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = TrackedVehicle(
                    tid, det['bbox'], det['ground_point'], det['class_name'], timestamp
                )

        # Handle unmatched tracks (temporarily lost or left scene)
        lost_ids = []
        for trk_idx, tid in enumerate(track_ids):
            if trk_idx not in matched_tracks:
                self.tracks[tid].lost_frames += 1
                if self.tracks[tid].lost_frames > self.max_lost:
                    lost_ids.append(tid)

        for tid in lost_ids:
            del self.tracks[tid]

        return self.tracks
