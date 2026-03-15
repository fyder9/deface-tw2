"""
Face Tracking Gap-Filler Module

Lightweight face tracking system for maintaining temporal consistency in video anonymization.
Fills detection gaps (1-10 frames) with continued blurring using last known box position.

Key Features:
- Operates ONLY on original frame pixel coordinates (detector-agnostic)
- Size-adaptive matching (normalized distance)
- Intentional box expansion during gaps (privacy feature)
- Simplified lifecycle (TTL only)
- Greedy matching (deterministic, auditable)
- No heavy dependencies (NumPy + OpenCV only)
"""

import numpy as np
from typing import List, Tuple


class Track:
    """
    Lightweight track representation for face tracking.

    Attributes:
        id: Unique track identifier
        box: [x1, y1, x2, y2] bounding box in original frame pixel coordinates
        original_box: Original box before any gap expansion (reference for expansion)
        lms: [10 values] landmarks (zeros for YOLO compatibility)
        score: Last detection confidence
        misses: Consecutive frames without detection
        last_seen_frame: Frame index of last detection/match
    """

    def __init__(self, track_id: int, box: np.ndarray, lms: np.ndarray,
                 score: float, frame_idx: int):
        """Initialize a new track."""
        self.id = track_id
        self.box = np.array(box[:4], dtype=np.float32)
        self.original_box = np.array(box[:4], dtype=np.float32)  # Reference for expansion
        self.lms = np.array(lms, dtype=np.float32) if lms is not None else np.zeros(10, dtype=np.float32)
        self.score = float(score)
        self.misses = 0
        self.last_seen_frame = frame_idx
        self.confirmation_count = 1  # Count of consecutive detections (confirmation window)

    def update(self, box: np.ndarray, lms: np.ndarray, score: float,
               frame_idx: int, alpha: float):
        """
        Update track with new detection using EMA smoothing.

        Args:
            box: Detection box [x1, y1, x2, y2]
            lms: Landmarks [10 values] or None
            score: Detection confidence
            frame_idx: Current frame index
            alpha: EMA smoothing factor (0-1)
        """
        # EMA smoothing on box coordinates
        new_box = np.array(box[:4], dtype=np.float32)
        self.box = alpha * new_box + (1 - alpha) * self.box
        self.original_box = new_box.copy()  # Update reference for expansion

        # Update landmarks (no smoothing - too unstable)
        if lms is not None and len(lms) > 0:
            self.lms = np.array(lms, dtype=np.float32)

        self.score = float(score)
        self.misses = 0
        self.last_seen_frame = frame_idx
        self.confirmation_count += 1  # Increment confirmation counter

    def mark_miss(self, frame_idx: int, expansion_rate: float = 0.05,
                  max_expansion: float = 1.20):
        """
        Increment miss counter and expand box for privacy.

        Args:
            frame_idx: Current frame index
            expansion_rate: Expansion factor per miss (e.g., 0.05 = 5%)
            max_expansion: Maximum cumulative expansion (e.g., 1.20 = 20%)
        """
        self.misses += 1

        # Expand box for privacy (over-blur > under-blur)
        # Get current box center and dimensions
        x1, y1, x2, y2 = self.box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Calculate cumulative expansion
        orig_x1, orig_y1, orig_x2, orig_y2 = self.original_box
        orig_w = orig_x2 - orig_x1
        orig_h = orig_y2 - orig_y1
        current_expansion = w / orig_w if orig_w > 0 else 1.0

        # Apply expansion if under max limit
        if current_expansion < max_expansion:
            expansion_factor = 1 + expansion_rate
            new_w = w * expansion_factor
            new_h = h * expansion_factor

            # Cap to max expansion
            if new_w / orig_w > max_expansion:
                new_w = orig_w * max_expansion
            if new_h / orig_h > max_expansion:
                new_h = orig_h * max_expansion

            # Update box with new dimensions
            self.box = np.array([
                cx - new_w / 2,
                cy - new_h / 2,
                cx + new_w / 2,
                cy + new_h / 2
            ], dtype=np.float32)

    def is_alive(self, ttl: int) -> bool:
        """Check if track is still active (within TTL)."""
        return self.misses <= ttl

    def is_confirmed(self, confirmation_window: int = 2) -> bool:
        """
        Check if track has been confirmed (seen in at least confirmation_window consecutive frames).

        Args:
            confirmation_window: Minimum consecutive detections required (default: 2)

        Returns:
            True if confirmation_count >= confirmation_window, False otherwise
        """
        return self.confirmation_count >= confirmation_window

    def to_detection(self, num_cols: int = 5) -> np.ndarray:
        """
        Convert track to detection format.

        Args:
            num_cols: Number of output columns (default 5 for [x1, y1, x2, y2, score]).
                     Extra columns are padded with zeros for compatibility with multi-column detectors.

        Returns:
            Detection array with shape (num_cols,) where extra columns are zeros.
        """
        det = np.concatenate([self.box, [self.score]])
        if num_cols > 5:
            # Pad with zeros for extra columns (e.g., class_id from CrowdHuman)
            padding = np.zeros(num_cols - 5, dtype=np.float32)
            det = np.concatenate([det, padding])
        return det


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1, box2: [x1, y1, x2, y2] format

    Returns:
        IoU value in [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection rectangle
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Check if boxes intersect
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def calculate_normalized_distance(det_box: np.ndarray,
                                  track_box: np.ndarray) -> float:
    """
    Calculate center distance normalized by track box diagonal.

    This makes matching scale-invariant (works for small and large faces).

    Args:
        det_box: Detection [x1, y1, x2, y2]
        track_box: Track [x1, y1, x2, y2]

    Returns:
        Normalized distance (unitless, typically 0-5)
    """
    # Centers
    cx_det = (det_box[0] + det_box[2]) / 2
    cy_det = (det_box[1] + det_box[3]) / 2

    cx_track = (track_box[0] + track_box[2]) / 2
    cy_track = (track_box[1] + track_box[3]) / 2

    # Euclidean distance between centers
    center_dist = np.sqrt((cx_det - cx_track)**2 + (cy_det - cy_track)**2)

    # Track box diagonal (normalization factor)
    w_track = track_box[2] - track_box[0]
    h_track = track_box[3] - track_box[1]
    diagonal = np.sqrt(w_track**2 + h_track**2)

    if diagonal < 1e-6:  # Avoid division by zero
        return float('inf')

    return center_dist / diagonal


def associate_detections_to_tracks(detections: np.ndarray,
                                   tracks: List[Track],
                                   iou_threshold: float,
                                   norm_dist_threshold: float) -> Tuple[List[Tuple], List[int], List[int]]:
    """
    Associate detections to existing tracks using IoU + normalized distance fallback.

    Uses greedy matching for simplicity and auditability.

    Args:
        detections: Nx5 array [x1, y1, x2, y2, score] in original frame pixel coordinates
        tracks: List of Track objects
        iou_threshold: Minimum IoU for match (e.g., 0.25)
        norm_dist_threshold: Maximum normalized distance for fallback (e.g., 1.5)

    Returns:
        matches: List of (det_idx, track_idx) tuples
        unmatched_dets: List of detection indices
        unmatched_tracks: List of track indices
    """
    if len(detections) == 0 or len(tracks) == 0:
        return [], list(range(len(detections))), list(range(len(tracks)))

    num_dets = len(detections)
    num_tracks = len(tracks)

    # Build cost matrix (IoU and distance)
    cost_matrix = np.zeros((num_dets, num_tracks), dtype=np.float32)
    dist_matrix = np.zeros((num_dets, num_tracks), dtype=np.float32)

    for d_idx in range(num_dets):
        det_box = detections[d_idx][:4]
        for t_idx in range(num_tracks):
            track_box = tracks[t_idx].box
            cost_matrix[d_idx, t_idx] = calculate_iou(det_box, track_box)
            dist_matrix[d_idx, t_idx] = calculate_normalized_distance(det_box, track_box)

    # Greedy matching: highest IoU first
    matches = []
    used_dets = set()
    used_tracks = set()

    # Flatten and sort by IoU (descending)
    candidates = []
    for d_idx in range(num_dets):
        for t_idx in range(num_tracks):
            iou = cost_matrix[d_idx, t_idx]
            norm_dist = dist_matrix[d_idx, t_idx]
            candidates.append((iou, norm_dist, d_idx, t_idx))

    candidates.sort(reverse=True, key=lambda x: x[0])  # Sort by IoU descending

    # Primary matching: IoU-based
    for iou, norm_dist, d_idx, t_idx in candidates:
        if d_idx in used_dets or t_idx in used_tracks:
            continue

        if iou >= iou_threshold:
            matches.append((d_idx, t_idx))
            used_dets.add(d_idx)
            used_tracks.add(t_idx)

    # Fallback matching: normalized distance-based for remaining pairs
    # Sort remaining candidates by normalized distance (ascending) for best fallback matching
    fallback_candidates = [c for c in candidates if c[2] not in used_dets and c[3] not in used_tracks]
    fallback_candidates.sort(key=lambda x: x[1])  # Sort by norm_dist ascending

    for iou, norm_dist, d_idx, t_idx in fallback_candidates:
        if d_idx in used_dets or t_idx in used_tracks:
            continue

        if norm_dist <= norm_dist_threshold:
            matches.append((d_idx, t_idx))
            used_dets.add(d_idx)
            used_tracks.add(t_idx)

    unmatched_dets = [i for i in range(num_dets) if i not in used_dets]
    unmatched_tracks = [i for i in range(num_tracks) if i not in used_tracks]

    return matches, unmatched_dets, unmatched_tracks


class FaceTracker:
    """
    Lightweight face tracker for filling detection gaps.

    Maintains temporal consistency in video anonymization by continuing to blur
    faces for 1-10 frames after detection loss, then re-associating when detected again.

    Configuration:
        iou_threshold: Minimum IoU for track association (default: 0.25)
        norm_dist_threshold: Max normalized distance fallback (default: 1.5)
        alpha: EMA smoothing factor for boxes (default: 0.65)
        ttl: Time-to-live for gap-filling (default: 10 frames)
        expansion_rate: Box expansion per miss (default: 0.05 = 5%)
        confirmation_window: Consecutive detections required before gap-fill activates (default: 2)
        debug: Enable debug logging (default: False)
    """

    def __init__(self,
                 iou_threshold: float = 0.25,
                 norm_dist_threshold: float = 1.5,
                 alpha: float = 0.65,
                 ttl: int = 10,
                 expansion_rate: float = 0.05,
                 confirmation_window: int = 2,
                 debug: bool = False):
        self.iou_threshold = iou_threshold
        self.norm_dist_threshold = norm_dist_threshold
        self.alpha = alpha
        self.ttl = ttl
        self.expansion_rate = expansion_rate
        self.confirmation_window = confirmation_window
        self.debug = debug

        self.tracks = []
        self.next_track_id = 0
        self.frame_idx = 0

        # Statistics for auditing
        self.stats = {
            'total_detections': 0,
            'total_tracks_created': 0,
            'total_tracks_deleted': 0,
            'frames_with_gaps_filled': 0,
            'total_gap_fills': 0
        }

    def update(self, detections: np.ndarray, landmarks: np.ndarray,
               frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update tracker with new detections.

        Args:
            detections: Nx5+ array [x1, y1, x2, y2, score, ...] in original frame pixel coordinates
                       Extra columns (e.g., class_id from CrowdHuman) are preserved
            landmarks: Nx10 array (can be zeros for YOLO or detectors without landmarks)
            frame_idx: Current frame number

        Returns:
            augmented_dets: Nx(5+) array including gap-filled tracks (5+ columns depending on input)
            augmented_lms: Nx10 array corresponding to augmented_dets
        """
        self.frame_idx = frame_idx
        self.stats['total_detections'] += len(detections)

        # Ensure landmarks array is proper shape
        if len(landmarks) == 0 and len(detections) > 0:
            landmarks = np.zeros((len(detections), 10), dtype=np.float32)

        # Step 1: Associate detections to existing tracks
        matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(
            detections, self.tracks, self.iou_threshold, self.norm_dist_threshold
        )

        if self.debug:
            print(f"[Track Frame {frame_idx}] Dets={len(detections)}, Tracks={len(self.tracks)}, "
                  f"Matches={len(matches)}, Unmatched_dets={len(unmatched_dets)}, "
                  f"Unmatched_tracks={len(unmatched_tracks)}")

        # Step 2: Update matched tracks
        for det_idx, track_idx in matches:
            det = detections[det_idx]
            lms = landmarks[det_idx] if len(landmarks) > det_idx else None
            self.tracks[track_idx].update(det[:4], lms, det[4], frame_idx, self.alpha)

        # Step 3: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            lms = landmarks[det_idx] if len(landmarks) > det_idx else None
            new_track = Track(self.next_track_id, det[:4], lms, det[4], frame_idx)
            self.tracks.append(new_track)
            self.next_track_id += 1
            self.stats['total_tracks_created'] += 1

        # Step 4: Mark unmatched tracks as missed (increments misses and expands box)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_miss(frame_idx, self.expansion_rate)

        # Step 5: Collect gap-filled tracks (within TTL, confirmed, and missed this frame)
        gap_filled_tracks = []
        gap_filled_indices = []
        for track_idx, track in enumerate(self.tracks):
            if (track.misses > 0 and track.is_alive(self.ttl) and
                    track.is_confirmed(self.confirmation_window)):
                gap_filled_tracks.append(track)
                gap_filled_indices.append(track_idx)

        if len(gap_filled_tracks) > 0:
            self.stats['frames_with_gaps_filled'] += 1
            self.stats['total_gap_fills'] += len(gap_filled_tracks)

            if self.debug:
                print(f"[Track Frame {frame_idx}] Gap-filling {len(gap_filled_tracks)} tracks: "
                      f"{[t.id for t in gap_filled_tracks]}")

        # Step 6: Delete expired tracks (misses > TTL)
        original_count = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.is_alive(self.ttl)]
        deleted_count = original_count - len(self.tracks)
        if deleted_count > 0:
            self.stats['total_tracks_deleted'] += deleted_count

        # Step 7: Build augmented detection array (original dets + gap-filled)
        augmented_dets = list(detections)
        augmented_lms = list(landmarks) if len(landmarks) > 0 else []

        # Determine column count from input detections (handles multi-column detectors like CrowdHuman)
        num_det_cols = detections.shape[1] if len(detections) > 0 else 5

        for track in gap_filled_tracks:
            gap_det = track.to_detection(num_cols=num_det_cols)
            augmented_dets.append(gap_det)
            augmented_lms.append(track.lms)

        # Convert to numpy arrays
        if len(augmented_dets) > 0:
            augmented_dets = np.array(augmented_dets, dtype=np.float32)
        else:
            augmented_dets = np.zeros((0, num_det_cols), dtype=np.float32)

        if len(augmented_lms) > 0:
            augmented_lms = np.array(augmented_lms, dtype=np.float32)
        else:
            augmented_lms = np.zeros((0, 10), dtype=np.float32)

        return augmented_dets, augmented_lms

    def get_stats(self) -> dict:
        """Return tracking statistics for auditing."""
        return self.stats.copy()
