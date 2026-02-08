"""
Unit tests for proximity search module
"""

import unittest
import numpy as np
from deface.proximity_search import (
    clamp_box, expand_box, calculate_iou, calculate_center_distance,
    calculate_box_diagonal, crop_roi_from_frame, map_roi_to_frame,
    validate_candidate, ProximitySearchConfig, ConfirmedFace
)


class TestHelperFunctions(unittest.TestCase):
    """Test basic helper functions"""

    def test_clamp_box_within_bounds(self):
        """Box within bounds should not change"""
        box = np.array([10, 20, 100, 120], dtype=np.float32)
        result = clamp_box(box, 640, 480)
        np.testing.assert_array_equal(result, box)

    def test_clamp_box_partially_out_of_bounds(self):
        """Partially out of bounds should clamp to edges"""
        box = np.array([-10, 10, 100, 120], dtype=np.float32)
        result = clamp_box(box, 640, 480)
        expected = np.array([0, 10, 100, 120], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_clamp_box_fully_out_of_bounds(self):
        """Partially out of bounds bottom-right should clamp to edges"""
        box = np.array([500, 630, 600, 700], dtype=np.float32)
        result = clamp_box(box, 640, 480)
        # Image: 640 height, 480 width. Max valid: x=479, y=639
        expected = np.array([500, 630, 479, 639], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_expand_box_1_8x(self):
        """Expand 100x120 box by 1.8x should give ~180x228"""
        box = np.array([500, 300, 600, 420], dtype=np.float32)
        result = expand_box(box, 1.8)
        # Original: w=100, h=120
        # Expanded: w*1.8=180, h*1.8=216
        # Padding: (180-100)/2=40 per side x, (216-120)/2=48 per side y
        # Result: [500-40, 300-48, 600+40, 420+48] = [460, 252, 640, 468]
        expected = np.array([460, 252, 640, 468], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=0)

    def test_expand_box_identity(self):
        """Expand by 1.0x should give same box"""
        box = np.array([100, 100, 200, 300], dtype=np.float32)
        result = expand_box(box, 1.0)
        np.testing.assert_array_almost_equal(result, box, decimal=5)

    def test_calculate_iou_perfect_overlap(self):
        """Perfect overlap should give IoU=1.0"""
        box1 = np.array([0, 0, 100, 100], dtype=np.float32)
        box2 = np.array([0, 0, 100, 100], dtype=np.float32)
        iou = calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0, places=5)

    def test_calculate_iou_no_overlap(self):
        """No overlap should give IoU=0.0"""
        box1 = np.array([0, 0, 100, 100], dtype=np.float32)
        box2 = np.array([200, 200, 300, 300], dtype=np.float32)
        iou = calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 0.0, places=5)

    def test_calculate_iou_partial_overlap(self):
        """Partial overlap should give intermediate value"""
        box1 = np.array([0, 0, 100, 100], dtype=np.float32)
        box2 = np.array([50, 50, 150, 150], dtype=np.float32)
        iou = calculate_iou(box1, box2)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500/17500 ≈ 0.1429
        self.assertAlmostEqual(iou, 2500/17500, places=4)

    def test_calculate_center_distance_same_box(self):
        """Distance between same boxes should be 0"""
        box = np.array([0, 0, 100, 100], dtype=np.float32)
        dist = calculate_center_distance(box, box)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_calculate_center_distance_offset(self):
        """Distance between offset boxes"""
        box1 = np.array([0, 0, 100, 100], dtype=np.float32)
        box2 = np.array([30, 40, 130, 140], dtype=np.float32)
        dist = calculate_center_distance(box1, box2)
        # Center 1: (50, 50)
        # Center 2: (80, 90)
        # Distance: sqrt(30^2 + 40^2) = sqrt(900 + 1600) = sqrt(2500) = 50
        self.assertAlmostEqual(dist, 50.0, places=5)

    def test_calculate_box_diagonal(self):
        """Test diagonal calculation"""
        box = np.array([0, 0, 300, 400], dtype=np.float32)
        diag = calculate_box_diagonal(box)
        # 3-4-5 triangle: diagonal = 500
        self.assertAlmostEqual(diag, 500.0, places=5)

    def test_crop_roi_from_frame(self):
        """Test ROI cropping"""
        frame = np.ones((100, 200, 3), dtype=np.uint8) * 128
        roi_box = np.array([50, 20, 150, 80], dtype=np.float32)
        roi_frame, x1, y1 = crop_roi_from_frame(frame, roi_box)

        self.assertEqual(roi_frame.shape, (60, 100, 3))
        self.assertEqual(x1, 50)
        self.assertEqual(y1, 20)

    def test_map_roi_to_frame_boxes(self):
        """Test coordinate mapping from ROI to frame"""
        # Detections in ROI coordinates
        dets_roi = np.array([[25, 30, 75, 90, 0.85]], dtype=np.float32)
        lms_roi = np.array([[30, 35, 40, 45, 50, 55, 60, 65, 70, 75]], dtype=np.float32)

        # ROI origin in frame
        roi_x1, roi_y1 = 460, 246

        dets_frame, lms_frame = map_roi_to_frame(dets_roi, lms_roi, roi_x1, roi_y1)

        # Expected: add roi origin to coordinates
        expected_dets = np.array([[485, 276, 535, 336, 0.85]], dtype=np.float32)
        expected_lms = np.array([[490, 281, 500, 291, 510, 301, 520, 311, 530, 321]], dtype=np.float32)

        np.testing.assert_array_almost_equal(dets_frame, expected_dets, decimal=4)
        np.testing.assert_array_almost_equal(lms_frame, expected_lms, decimal=4)

    def test_map_roi_to_frame_empty(self):
        """Empty detections should remain empty"""
        dets_roi = np.empty((0, 5), dtype=np.float32)
        lms_roi = np.empty((0, 10), dtype=np.float32)

        dets_frame, lms_frame = map_roi_to_frame(dets_roi, lms_roi, 100, 100)

        self.assertEqual(dets_frame.shape, (0, 5))
        self.assertEqual(lms_frame.shape, (0, 10))


class TestValidation(unittest.TestCase):
    """Test validation gates"""

    def test_validate_score_gate(self):
        """Low score should fail validation"""
        config = ProximitySearchConfig(proximity_thresh=0.5)
        candidate_box = np.array([100, 100, 200, 200], dtype=np.float32)
        last_confirmed_box = np.array([100, 100, 200, 200], dtype=np.float32)

        # Perfect match but low score
        result = validate_candidate(candidate_box, 0.4, last_confirmed_box, config)
        self.assertFalse(result)

    def test_validate_score_gate_pass(self):
        """High score should pass score gate"""
        config = ProximitySearchConfig(proximity_thresh=0.5)
        candidate_box = np.array([100, 100, 200, 200], dtype=np.float32)
        last_confirmed_box = np.array([100, 100, 200, 200], dtype=np.float32)

        result = validate_candidate(candidate_box, 0.8, last_confirmed_box, config)
        self.assertTrue(result)

    def test_validate_position_iou_gate(self):
        """High IoU should pass position gate"""
        config = ProximitySearchConfig(
            proximity_thresh=0.3,
            proximity_iou=0.15,
            proximity_dist=1.2
        )
        candidate_box = np.array([0, 0, 100, 100], dtype=np.float32)
        last_confirmed_box = np.array([0, 0, 100, 100], dtype=np.float32)

        result = validate_candidate(candidate_box, 0.8, last_confirmed_box, config)
        self.assertTrue(result)

    def test_validate_position_distance_gate(self):
        """Close center distance should pass position gate"""
        config = ProximitySearchConfig(
            proximity_thresh=0.3,
            proximity_iou=0.15,
            proximity_dist=1.2
        )
        # Offset by 30 pixels horizontally
        candidate_box = np.array([30, 0, 130, 100], dtype=np.float32)
        last_confirmed_box = np.array([0, 0, 100, 100], dtype=np.float32)

        # Distance = 30, diagonal = sqrt(100^2 + 100^2) ≈ 141.4
        # normalized_dist = 30 / 141.4 ≈ 0.21
        # Should pass with proximity_dist=1.2
        result = validate_candidate(candidate_box, 0.8, last_confirmed_box, config)
        self.assertTrue(result)

    def test_validate_position_distance_fail(self):
        """Far center distance should fail position gate"""
        config = ProximitySearchConfig(
            proximity_thresh=0.3,
            proximity_iou=0.0,  # No IoU overlap
            proximity_dist=0.5  # Strict distance threshold
        )
        # Very far offset: candidate center (250, 250) vs last (50, 50)
        # Distance = sqrt(200^2 + 200^2) = 282.8
        # Diagonal of last = 141.4, normalized = 282.8/141.4 = 2.0
        # With proximity_dist=0.5, this should fail
        candidate_box = np.array([200, 200, 300, 300], dtype=np.float32)
        last_confirmed_box = np.array([0, 0, 100, 100], dtype=np.float32)

        # Actually this passes with default iou gate. Let's use a stricter config:
        config.proximity_iou = 0.5  # Require 50% overlap (unlikely)
        config.proximity_dist = 0.2  # Very strict distance
        result = validate_candidate(candidate_box, 0.8, last_confirmed_box, config)
        self.assertFalse(result)


class TestConfirmedFace(unittest.TestCase):
    """Test ConfirmedFace state container"""

    def test_confirmed_face_creation(self):
        """Should create face with all attributes"""
        box = np.array([100, 100, 200, 200], dtype=np.float32)
        lms = np.arange(10, dtype=np.float32)

        face = ConfirmedFace(
            id=1,
            last_confirmed_box=box,
            last_confirmed_score=0.95,
            last_confirmed_frame_idx=10,
            last_confirmed_lms=lms
        )

        self.assertEqual(face.id, 1)
        np.testing.assert_array_equal(face.last_confirmed_box, box)
        self.assertEqual(face.last_confirmed_score, 0.95)
        self.assertEqual(face.last_confirmed_frame_idx, 10)
        np.testing.assert_array_equal(face.last_confirmed_lms, lms)

    def test_confirmed_face_default_landmarks(self):
        """Should use default zero landmarks if not provided"""
        box = np.array([100, 100, 200, 200], dtype=np.float32)

        face = ConfirmedFace(
            id=1,
            last_confirmed_box=box,
            last_confirmed_score=0.95,
            last_confirmed_frame_idx=10
        )

        self.assertEqual(len(face.last_confirmed_lms), 10)
        np.testing.assert_array_equal(face.last_confirmed_lms, np.zeros(10))


if __name__ == '__main__':
    unittest.main()
