"""
Proximity Search Module - Targeted re-detection around last confirmed face positions

This module implements ROI-based re-detection to reacquire faces that were detected in
prior frames but missed by full-frame detection. It uses last confirmed positions to seed
ROI searches, complementing full-frame detection and tracking.

Key Design Principles:
- ONLY uses last_confirmed_box from real detector outputs to seed ROI searches
- NEVER uses gap-filled/predicted boxes as seeds
- Works with all detectors (CenterFace, SCRFD, YOLO) via uniform interface
- Operates entirely in original frame pixel coordinates
- Conservative validation gates ensure high precision
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# Helper Functions
# =============================================================================

def clamp_box(box: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """
    Clamp bounding box to image boundaries.

    Args:
        box: [x1, y1, x2, y2]
        img_h: image height
        img_w: image width

    Returns:
        Clamped box in same format
    """
    x1, y1, x2, y2 = box
    x1 = max(0, int(np.round(x1)))
    y1 = max(0, int(np.round(y1)))
    x2 = min(img_w - 1, int(np.round(x2)))
    y2 = min(img_h - 1, int(np.round(y2)))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def expand_box(box: np.ndarray, expand_factor: float) -> np.ndarray:
    """
    Expand bounding box symmetrically by expand_factor.

    For expand_factor=1.8:
        - Original 100x120 face → 180x228 ROI
        - Padding = 40x54 on each side

    Args:
        box: [x1, y1, x2, y2]
        expand_factor: Expansion multiplier (e.g., 1.8)

    Returns:
        Expanded box [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1

    # Calculate padding needed
    new_w, new_h = w * expand_factor, h * expand_factor
    pad_x, pad_y = (new_w - w) / 2, (new_h - h) / 2

    return np.array([
        x1 - pad_x,
        y1 - pad_y,
        x2 + pad_x,
        y2 + pad_y
    ], dtype=np.float32)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 < xi1 or yi2 < yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return float(inter_area / union_area)


def calculate_center_distance(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between box centers.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        Euclidean distance in pixels
    """
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def calculate_box_diagonal(box: np.ndarray) -> float:
    """
    Calculate diagonal length of bounding box.

    For normalized distance comparisons: distance <= proximity_dist * diagonal

    Args:
        box: [x1, y1, x2, y2]

    Returns:
        Diagonal length in pixels
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    return float(np.sqrt(w ** 2 + h ** 2))


def crop_roi_from_frame(frame: np.ndarray, roi_box: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Crop ROI from frame and return ROI frame plus origin coordinates.

    Args:
        frame: Frame array (H, W, 3)
        roi_box: [x1, y1, x2, y2] in frame coordinates

    Returns:
        (roi_frame, x1_roi, y1_roi): Cropped frame and top-left origin
    """
    x1, y1, x2, y2 = roi_box.astype(int)
    roi_frame = frame[y1:y2, x1:x2]
    return roi_frame, x1, y1


def map_roi_to_frame(dets_roi: np.ndarray, lms_roi: np.ndarray,
                     roi_x1: int, roi_y1: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert detections from ROI-local coordinates to frame coordinates.

    Boxes are in ROI-local (0, 0) -> (roi_w, roi_h).
    Landmarks are in ROI-local coordinates.

    Args:
        dets_roi: (N, 5) array [x1, y1, x2, y2, score] in ROI-local coords
        lms_roi: (N, 10) array [x1, y1, ..., x5, y5] in ROI-local coords
        roi_x1, roi_y1: ROI origin in frame coordinates

    Returns:
        (dets_frame, lms_frame): Converted to frame coordinates
    """
    if len(dets_roi) == 0:
        return dets_roi, lms_roi

    dets_frame = dets_roi.copy()
    # Add roi origin to x coordinates (indices 0, 2)
    dets_frame[:, [0, 2]] += roi_x1
    # Add roi origin to y coordinates (indices 1, 3)
    dets_frame[:, [1, 3]] += roi_y1

    lms_frame = lms_roi.copy()
    if len(lms_frame) > 0:
        # Landmarks: 10 values = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        # Add roi origin to all x coordinates (even indices 0, 2, 4, 6, 8)
        lms_frame[:, [0, 2, 4, 6, 8]] += roi_x1
        # Add roi origin to all y coordinates (odd indices 1, 3, 5, 7, 9)
        lms_frame[:, [1, 3, 5, 7, 9]] += roi_y1

    return dets_frame, lms_frame


# =============================================================================
# Detector Interface
# =============================================================================

def run_detector_on_roi(detector: Any, frame: np.ndarray, roi_box: np.ndarray,
                        threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run detector on ROI and convert returned coordinates to frame coordinates.

    This function handles the coordinate mapping transparently. Detectors are
    called with a cropped ROI frame and return ROI-local coordinates, which
    are converted back to frame coordinates.

    Args:
        detector: Detector instance with __call__(frame, threshold) interface
        frame: Original full frame (H, W, 3)
        roi_box: [x1, y1, x2, y2] in frame coordinates
        threshold: Detection confidence threshold

    Returns:
        (dets_frame, lms_frame): Detections in frame coordinates
            dets_frame: (N, 5) [x1, y1, x2, y2, score]
            lms_frame: (N, 10) [x1, y1, ..., x5, y5]
    """
    # Crop ROI from frame
    roi_frame, roi_x1, roi_y1 = crop_roi_from_frame(frame, roi_box)

    # Detect on ROI (returns ROI-local coordinates)
    dets_roi, lms_roi = detector(roi_frame, threshold=threshold)

    # Map to frame coordinates
    dets_frame, lms_frame = map_roi_to_frame(dets_roi, lms_roi, roi_x1, roi_y1)

    return dets_frame, lms_frame


# =============================================================================
# Validation
# =============================================================================

def validate_candidate(candidate_box: np.ndarray, candidate_score: float,
                       last_confirmed_box: np.ndarray,
                       config: 'ProximitySearchConfig') -> bool:
    """
    Multi-gate validation for proximity search candidates.

    Gates (all must pass):
        1. Score: candidate_score >= proximity_thresh
        2. Position: IoU >= proximity_iou OR center_distance <= proximity_dist * diagonal

    Args:
        candidate_box: [x1, y1, x2, y2] detected in ROI
        candidate_score: Detection confidence
        last_confirmed_box: [x1, y1, x2, y2] from last real detection
        config: ProximitySearchConfig with validation thresholds

    Returns:
        True if candidate passes all gates, False otherwise
    """
    # Gate 1: Score threshold
    if candidate_score < config.proximity_thresh:
        return False

    # Gate 2: Position validation (IoU OR distance)
    iou = calculate_iou(candidate_box, last_confirmed_box)
    if iou >= config.proximity_iou:
        return True  # Strong overlap is good enough

    center_dist = calculate_center_distance(candidate_box, last_confirmed_box)
    diagonal = calculate_box_diagonal(last_confirmed_box)
    if center_dist / diagonal <= config.proximity_dist:
        return True  # Close enough to confirmed position

    # Position validation failed
    return False


# =============================================================================
# Configuration & State
# =============================================================================

@dataclass
class ProximitySearchConfig:
    """Configuration for proximity search behavior."""
    proximity_ttl: int = 10
    proximity_expand: float = 1.8
    proximity_thresh: float = 0.3
    proximity_iou: float = 0.15
    proximity_dist: float = 1.2
    proximity_iou_assoc: float = 0.3
    debug: bool = False


@dataclass
class ConfirmedFace:
    """
    Tracks last confirmed detection for a face.

    Attributes:
        id: Unique face identifier
        last_confirmed_box: [x1, y1, x2, y2] from last real detection
        last_confirmed_score: Detection confidence from last real detection
        last_confirmed_frame_idx: Frame index of last confirmation
        last_confirmed_lms: Landmarks [10 values] from last real detection
    """
    id: int
    last_confirmed_box: np.ndarray  # [x1, y1, x2, y2]
    last_confirmed_score: float
    last_confirmed_frame_idx: int
    last_confirmed_lms: np.ndarray = field(default_factory=lambda: np.zeros(10))


# =============================================================================
# Proximity Search Manager
# =============================================================================

class ProximitySearchManager:
    """
    Manages proximity search state and orchestrates ROI-based re-detection.

    This class maintains a history of confirmed face detections and attempts to
    reacquire faces that disappear by running additional detector passes on ROIs
    around their last confirmed positions.

    Configuration parameters control:
        - proximity_ttl: Max frames since last confirmation to search (default: 10)
        - proximity_expand: ROI expansion factor (default: 1.8)
        - proximity_thresh: Score threshold for candidates (default: 0.3)
        - proximity_iou: Min IoU for position validation (default: 0.15)
        - proximity_dist: Max normalized center distance (default: 1.2)
        - debug: Enable debug logging (default: False)
    """

    def __init__(self, detector: Any, config: ProximitySearchConfig):
        """
        Initialize proximity search manager.

        Args:
            detector: Detector instance with __call__(frame, threshold) interface
            config: ProximitySearchConfig with all parameters
        """
        self.detector = detector
        self.config = config
        self.confirmed_faces: Dict[int, ConfirmedFace] = {}
        self.next_face_id = 0
        self.stats = {
            'total_roi_searches': 0,
            'successful_reacquisitions': 0,
            'failed_validations': 0
        }

    def update_confirmed_detections(self, dets: np.ndarray, lms: np.ndarray,
                                     frame_idx: int) -> None:
        """
        Update confirmed face states from current frame's detector output.

        Associates current detections to existing confirmed faces using greedy IoU
        matching. Updates existing faces or creates new ones.

        Args:
            dets: (N, 5) array [x1, y1, x2, y2, score] from detector
            lms: (N, 10) array [landmarks] from detector
            frame_idx: Current frame index
        """
        if len(dets) == 0:
            return

        # Build IoU matrix: dets × confirmed_faces
        confirmed_list = list(self.confirmed_faces.values())
        iou_matrix = np.zeros((len(dets), len(confirmed_list)))

        for d_idx, det in enumerate(dets):
            for f_idx, face in enumerate(confirmed_list):
                iou_matrix[d_idx, f_idx] = calculate_iou(det[:4], face.last_confirmed_box)

        # Greedy matching: highest IoU first
        matches = []
        used_dets = set()
        used_faces = set()

        # Create (iou, det_idx, face_idx) tuples sorted by IoU descending
        candidates = []
        for d_idx in range(len(dets)):
            for f_idx in range(len(confirmed_list)):
                candidates.append((iou_matrix[d_idx, f_idx], d_idx, f_idx))
        candidates.sort(reverse=True)

        # Greedy assignment
        for iou, d_idx, f_idx in candidates:
            if d_idx in used_dets or f_idx in used_faces:
                continue
            if iou >= self.config.proximity_iou_assoc:  # Minimum IoU for association
                matches.append((d_idx, f_idx))
                used_dets.add(d_idx)
                used_faces.add(f_idx)

        # Update matched faces
        for d_idx, f_idx in matches:
            face = confirmed_list[f_idx]
            face.last_confirmed_box = dets[d_idx, :4].copy()
            face.last_confirmed_score = float(dets[d_idx, 4])
            face.last_confirmed_frame_idx = frame_idx
            if len(lms) > d_idx:
                face.last_confirmed_lms = lms[d_idx].copy()

        # Create new faces for unmatched detections
        for d_idx in range(len(dets)):
            if d_idx not in used_dets:
                new_face = ConfirmedFace(
                    id=self.next_face_id,
                    last_confirmed_box=dets[d_idx, :4].copy(),
                    last_confirmed_score=float(dets[d_idx, 4]),
                    last_confirmed_frame_idx=frame_idx,
                    last_confirmed_lms=lms[d_idx].copy() if len(lms) > d_idx else np.zeros(10)
                )
                self.confirmed_faces[self.next_face_id] = new_face
                self.next_face_id += 1

        # Delete stale faces (not seen in > 2*proximity_ttl frames)
        stale_threshold = frame_idx - 2 * self.config.proximity_ttl
        self.confirmed_faces = {
            fid: face for fid, face in self.confirmed_faces.items()
            if face.last_confirmed_frame_idx > stale_threshold
        }

    def search_missing_faces(self, frame: np.ndarray, current_dets: np.ndarray,
                             frame_idx: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        For confirmed faces not matched in current_dets, attempt ROI-based reacquisition.

        Iterates through confirmed faces not found in current frame and runs proximity
        searches using their last confirmed boxes as seeds.

        Args:
            frame: Current frame (H, W, 3)
            current_dets: (M, 5) detections from full-frame detector (may be empty)
            frame_idx: Current frame index
            threshold: Detection threshold for full-frame detector

        Returns:
            (reacquired_dets, reacquired_lms): Successfully reacquired detections
                reacquired_dets: (K, 5) [x1, y1, x2, y2, score]
                reacquired_lms: (K, 10) [landmarks]
        """
        reacquired_dets_list = []
        reacquired_lms_list = []
        img_h, img_w = frame.shape[:2]

        # Find matched faces (those in current_dets)
        matched_face_ids = set()
        if len(current_dets) > 0:
            confirmed_list = list(self.confirmed_faces.values())
            iou_matrix = np.zeros((len(current_dets), len(confirmed_list)))

            for d_idx, det in enumerate(current_dets):
                for f_idx, face in enumerate(confirmed_list):
                    iou_matrix[d_idx, f_idx] = calculate_iou(det[:4], face.last_confirmed_box)

            # Mark matched faces
            for d_idx in range(len(current_dets)):
                for f_idx, face in enumerate(confirmed_list):
                    if iou_matrix[d_idx, f_idx] >= self.config.proximity_iou_assoc:
                        matched_face_ids.add(face.id)

        # Search for missing faces
        for face_id, face in self.confirmed_faces.items():
            # Skip if already matched in current frame
            if face_id in matched_face_ids:
                continue

            # Gate 1: Age filter
            frame_gap = frame_idx - face.last_confirmed_frame_idx
            if frame_gap > self.config.proximity_ttl:
                continue

            # Expand and clamp ROI
            roi_box = expand_box(face.last_confirmed_box, self.config.proximity_expand)
            roi_box = clamp_box(roi_box, img_h, img_w)

            # Run proximity search
            self.stats['total_roi_searches'] += 1
            dets_roi, lms_roi = run_detector_on_roi(
                self.detector, frame, roi_box, threshold
            )

            if len(dets_roi) == 0:
                continue

            # Validate candidates
            for d_idx, det in enumerate(dets_roi):
                if validate_candidate(det[:4], det[4], face.last_confirmed_box, self.config):
                    reacquired_dets_list.append(det)
                    if len(lms_roi) > d_idx:
                        reacquired_lms_list.append(lms_roi[d_idx])
                    else:
                        reacquired_lms_list.append(np.zeros(10))
                    self.stats['successful_reacquisitions'] += 1

                    if self.config.debug:
                        print(f"[Proximity Frame {frame_idx}] Reacquired face_id={face_id} "
                              f"at [{det[0]:.0f}, {det[1]:.0f}, {det[2]:.0f}, {det[3]:.0f}] "
                              f"(score: {det[4]:.2f})")
                    break  # Only use first valid candidate per face
                else:
                    self.stats['failed_validations'] += 1

        # Stack results
        if len(reacquired_dets_list) > 0:
            reacquired_dets = np.vstack(reacquired_dets_list)
            reacquired_lms = np.vstack(reacquired_lms_list)
        else:
            reacquired_dets = np.empty((0, 5), dtype=np.float32)
            reacquired_lms = np.empty((0, 10), dtype=np.float32)

        return reacquired_dets, reacquired_lms

    def get_stats(self) -> Dict[str, int]:
        """Return statistics for debugging/auditing."""
        return self.stats.copy()
