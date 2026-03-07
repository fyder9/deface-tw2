"""
CrowdHuman YOLOv5m detector for person/head detection in dense crowds.

Model: CrowdHuman-trained YOLOv5m (80 MB ONNX)
Input: RGB image, letterboxed to model input size (typically 640x640)
Output: Person/head bounding boxes with confidence scores

TEST COMMAND:
    deface video.mp4 --detector crowdhuman --thresh 0.4 --scores --preview

Expected startup output:
    [CrowdHuman-YOLOv5] Model I/O Introspection:
      Input: images, shape=[1, 3, 640, 640], type=tensor(float)
      Output: output, shape=[1, 25200, 85], type=tensor(float)
    [CrowdHuman-YOLOv5] Available providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    [CrowdHuman-YOLOv5] Running on: CUDAExecutionProvider

Model source:
    External ONNX model, not versioned in git. Must be manually placed in /models directory.
    The model file (crowdhuman_yolov5m.onnx) should be exported from a CrowdHuman-trained YOLOv5m checkpoint.
"""

import os
import numpy as np
import cv2
import onnxruntime as ort


class CrowdHumanYOLOv5Detector:
    """ONNX detector for CrowdHuman-trained YOLOv5m model."""

    def __init__(self, model_path, device='auto', override_execution_provider=None):
        """
        Initialize CrowdHuman YOLOv5m detector.

        Args:
            model_path (str): Path to crowdhuman_yolov5m.onnx model file
            device (str): Device selection strategy ('auto', 'cuda', 'cpu'). Default: 'auto'
            override_execution_provider (str): Force specific provider ('CUDAExecutionProvider',
                'DirectMLExecutionProvider', 'CPUExecutionProvider'). If set, ignores device parameter.

        Raises:
            FileNotFoundError: If model file does not exist
            RuntimeError: If no suitable execution provider found
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'Model file not found: {model_path}')

        # Determine execution providers
        available_providers = ort.get_available_providers()
        print(f'[CrowdHuman-YOLOv5] Available providers: {available_providers}')

        # Provider selection logic (matches YOLODetector pattern)
        if override_execution_provider:
            if override_execution_provider not in available_providers:
                raise RuntimeError(
                    f'Requested provider {override_execution_provider} not available. '
                    f'Available: {available_providers}'
                )
            providers = [override_execution_provider]
        else:
            if device == 'cuda' or device == 'auto':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            # Filter to available only
            providers = [p for p in providers if p in available_providers]

        if not providers:
            raise RuntimeError('No suitable execution providers found')

        print(f'[CrowdHuman-YOLOv5] Selected providers: {providers}')

        # Create session
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, sess_opts, providers=providers)

        print(f'[CrowdHuman-YOLOv5] Running on: {self.sess.get_providers()}')

        # Model introspection (print once at init, don't hardcode)
        print('[CrowdHuman-YOLOv5] Model I/O Introspection:')
        for inp in self.sess.get_inputs():
            print(f'  Input: {inp.name}, shape={inp.shape}, type={inp.type}')
        for out in self.sess.get_outputs():
            print(f'  Output: {out.name}, shape={out.shape}, type={out.type}')

        # Extract metadata
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [out.name for out in self.sess.get_outputs()]

        # Determine input size (handle dynamic shapes)
        input_shape = self.sess.get_inputs()[0].shape
        if isinstance(input_shape[2], int) and input_shape[2] > 0:
            self.input_size = input_shape[2]  # Static shape
        else:
            self.input_size = 640  # Safe default for YOLOv5
            print(f'[CrowdHuman-YOLOv5] Dynamic input shape, using default: {self.input_size}')

        # NMS parameters
        self.nms_iou = 0.45
        self.score_thresh = 0.3  # Default, overridden by __call__ threshold arg

    def ensure_rgb(self, img):
        """
        Convert grayscale or RGBA to RGB.

        Args:
            img (np.ndarray): Input image in BGR or grayscale

        Returns:
            np.ndarray: RGB image (or already RGB if input was RGB)
        """
        if len(img.shape) == 2:
            # Grayscale -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            # RGBA -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.shape[2] == 3:
            # BGR -> RGB (OpenCV default is BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preprocess(self, frame):
        """
        Letterbox preprocessing with aspect ratio preservation.

        YOLOv5 convention: letterbox with gray padding (value=114).

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            tuple: (input_blob, resize_ratio, pad_w, pad_h)
                - input_blob: (1, 3, H, W) float32 in [0, 1], NCHW format, RGB
                - resize_ratio: scale factor for coordinate restoration
                - pad_w, pad_h: padding offsets (pixels)
        """
        h, w = frame.shape[:2]

        # Compute scale to fit input_size while preserving aspect ratio
        resize_ratio = min(self.input_size / w, self.input_size / h)
        new_w = int(w * resize_ratio)
        new_h = int(h * resize_ratio)

        # Resize image
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox with gray padding (YOLOv5 standard: 114)
        pad_w = (self.input_size - new_w) // 2
        pad_h = (self.input_size - new_h) // 2
        letterboxed = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        letterboxed[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        # Convert BGR -> RGB and normalize to [0, 1]
        letterboxed_rgb = self.ensure_rgb(letterboxed).astype(np.float32)
        letterboxed_rgb = letterboxed_rgb / 255.0

        # Convert to NCHW: (H, W, C) -> (1, C, H, W)
        input_blob = np.transpose(letterboxed_rgb, (2, 0, 1))
        input_blob = np.expand_dims(input_blob, axis=0).astype(np.float32)

        return input_blob, resize_ratio, pad_w, pad_h

    def infer(self, frame):
        """
        Run inference on frame.

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            tuple: (outputs, h, w, resize_ratio, pad_w, pad_h)
        """
        frame = self.ensure_rgb(frame)
        h, w = frame.shape[:2]

        input_blob, resize_ratio, pad_w, pad_h = self.preprocess(frame)
        outputs = self.sess.run(self.output_names, {self.input_name: input_blob})

        return outputs, h, w, resize_ratio, pad_w, pad_h

    def decode_outputs(self, outputs, org_h, org_w, resize_ratio, pad_w, pad_h, score_thresh=None):
        """
        Decode YOLOv5 model outputs to pixel-space bounding boxes.

        YOLOv5 output formats supported:
            1. Standard (85 values): [x_center, y_center, width, height, objectness, class_prob_0, ..., class_prob_79]
               - objectness: probability of object presence
               - class_probs: per-class probabilities (80 classes for COCO)
               - confidence = objectness * max(class_probs)

            2. CrowdHuman (7 values): [x_center, y_center, width, height, confidence, class_id, class_prob]
               - confidence: already-computed confidence score
               - class_id, class_prob: additional metadata

            3. Simplified (6 values): [x_center, y_center, width, height, confidence, class_id]
               - confidence: already-computed confidence score

        Coordinate restoration:
            1. Predictions are in letterbox space (0 to input_size pixels)
            2. Subtract padding (pad_w, pad_h) to get resized-frame space
            3. Divide by resize_ratio to get original-frame space
            4. Clamp to frame bounds [0, org_h] × [0, org_w]

        Args:
            outputs (list): Model outputs from inference
            org_h, org_w (int): Original frame height, width
            resize_ratio (float): Scale factor used in preprocessing
            pad_w, pad_h (int): Padding offsets (pixels)
            score_thresh (float): Score threshold for detection. If None, uses self.score_thresh

        Returns:
            tuple: (pixel_boxes, pixel_lms)
                - pixel_boxes: list of [x1, y1, x2, y2, score, class_id] in pixel coordinates (class_id appended for CrowdHuman analysis)
                - pixel_lms: empty list (no landmarks for person detector)
        """
        pixel_boxes = []
        pixel_lms = []  # Always empty for person detector

        thresh = self.score_thresh if score_thresh is None else float(score_thresh)

        output = outputs[0]  # First (usually only) output
        predictions = output[0]  # Remove batch dimension
        num_preds, num_values = predictions.shape

        if num_values == 85:
            # Standard YOLOv5 format: [x, y, w, h, objectness, class_probs(80)]
            for pred in predictions:
                x_center, y_center, w, h = pred[0:4]
                objectness = pred[4]
                class_probs = pred[5:85]
                class_id = int(np.argmax(class_probs))
                score = objectness * np.max(class_probs)

                if score < thresh:
                    continue

                # Convert center-xywh to corner coordinates (letterbox space)
                x1_lb = x_center - w / 2.0
                y1_lb = y_center - h / 2.0
                x2_lb = x_center + w / 2.0
                y2_lb = y_center + h / 2.0

                # Remove letterbox padding and scale to original frame
                x1 = (x1_lb - pad_w) / resize_ratio
                y1 = (y1_lb - pad_h) / resize_ratio
                x2 = (x2_lb - pad_w) / resize_ratio
                y2 = (y2_lb - pad_h) / resize_ratio

                # Clamp to frame bounds
                x1 = max(0.0, min(x1, float(org_w - 1)))
                y1 = max(0.0, min(y1, float(org_h - 1)))
                x2 = max(0.0, min(x2, float(org_w - 1)))
                y2 = max(0.0, min(y2, float(org_h - 1)))

                if x2 <= x1 or y2 <= y1:
                    continue

                pixel_boxes.append([x1, y1, x2, y2, float(score), float(class_id)])

        elif num_values == 7:
            # CrowdHuman-specific format: [x, y, w, h, confidence, class_id, ???]
            # 7-value format appears to be: [x_center, y_center, w, h, conf, class_id, class_prob]
            for pred in predictions:
                x_center, y_center, w, h = pred[0:4]
                score = pred[4]
                class_id = int(pred[5])  # Extract class_id from position 5

                if score < thresh:
                    continue

                # Convert center-xywh to corner coordinates (letterbox space)
                x1_lb = x_center - w / 2.0
                y1_lb = y_center - h / 2.0
                x2_lb = x_center + w / 2.0
                y2_lb = y_center + h / 2.0

                # Remove letterbox padding and scale to original frame
                x1 = (x1_lb - pad_w) / resize_ratio
                y1 = (y1_lb - pad_h) / resize_ratio
                x2 = (x2_lb - pad_w) / resize_ratio
                y2 = (y2_lb - pad_h) / resize_ratio

                # Clamp to frame bounds
                x1 = max(0.0, min(x1, float(org_w - 1)))
                y1 = max(0.0, min(y1, float(org_h - 1)))
                x2 = max(0.0, min(x2, float(org_w - 1)))
                y2 = max(0.0, min(y2, float(org_h - 1)))

                if x2 <= x1 or y2 <= y1:
                    continue

                pixel_boxes.append([x1, y1, x2, y2, float(score), float(class_id)])

        elif num_values == 6:
            # Simplified format: [x, y, w, h, confidence, class_id]
            for pred in predictions:
                x_center, y_center, w, h = pred[0:4]
                score = pred[4]
                class_id = int(pred[5])  # Extract class_id from position 5

                if score < thresh:
                    continue

                # Convert center-xywh to corner coordinates (letterbox space)
                x1_lb = x_center - w / 2.0
                y1_lb = y_center - h / 2.0
                x2_lb = x_center + w / 2.0
                y2_lb = y_center + h / 2.0

                # Remove letterbox padding and scale to original frame
                x1 = (x1_lb - pad_w) / resize_ratio
                y1 = (y1_lb - pad_h) / resize_ratio
                x2 = (x2_lb - pad_w) / resize_ratio
                y2 = (y2_lb - pad_h) / resize_ratio

                # Clamp to frame bounds
                x1 = max(0.0, min(x1, float(org_w - 1)))
                y1 = max(0.0, min(y1, float(org_h - 1)))
                x2 = max(0.0, min(x2, float(org_w - 1)))
                y2 = max(0.0, min(y2, float(org_h - 1)))

                if x2 <= x1 or y2 <= y1:
                    continue

                pixel_boxes.append([x1, y1, x2, y2, float(score), float(class_id)])

        else:
            raise RuntimeError(
                f'[CrowdHuman-YOLOv5] Unexpected output format: shape={output.shape}. '
                f'Expected (1, num_predictions, 85), (1, num_predictions, 7), or (1, num_predictions, 6), '
                f'but got (1, {num_preds}, {num_values}).'
            )

        return pixel_boxes, pixel_lms

    def nms(self, pixel_boxes, pixel_lms, iou_threshold=None):
        """
        Apply non-maximum suppression to detections.

        Args:
            pixel_boxes (list): Bounding boxes [x1, y1, x2, y2, score]
            pixel_lms (list): Landmarks (empty for person detector)
            iou_threshold (float): NMS IoU threshold. If None, uses self.nms_iou

        Returns:
            tuple: (kept_boxes, kept_lms) after NMS
        """
        if not pixel_boxes:
            return [], []

        iou_thresh = self.nms_iou if iou_threshold is None else float(iou_threshold)

        # Convert boxes to OpenCV format: [x, y, w, h]
        boxes = []
        scores = []
        for box in pixel_boxes:
            x1, y1, x2, y2, score = box[:5]  # Extract first 5 values (ignore class_id)
            w = x2 - x1
            h = y2 - y1
            boxes.append([x1, y1, w, h])
            scores.append(score)

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        # Run NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou_thresh)

        # Convert indices to list and extract kept boxes
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        else:
            indices = indices if indices else []

        kept_boxes = [pixel_boxes[i] for i in indices]
        kept_lms = [pixel_lms[i] for i in indices] if pixel_lms else []

        return kept_boxes, kept_lms

    def __call__(self, frame, threshold=0.5):
        """
        CenterFace-compatible callable interface for face detection.

        Args:
            frame (np.ndarray): Input frame in BGR format
            threshold (float): Detection confidence threshold

        Returns:
            tuple: (dets, lms)
                - dets: (N, 6) array [x1, y1, x2, y2, score, class_id] in pixel coordinates
                  Note: class_id appended for CrowdHuman head/body differentiation
                - lms: (N, 10) array of zeros (no landmarks for person detector)
        """
        outputs, org_h, org_w, resize_ratio, pad_w, pad_h = self.infer(frame)

        pixel_boxes, pixel_lms = self.decode_outputs(
            outputs, org_h, org_w, resize_ratio, pad_w, pad_h,
            score_thresh=threshold
        )

        kept_boxes, kept_lms = self.nms(pixel_boxes, pixel_lms)

        # Convert to numpy arrays with correct shapes
        # CrowdHuman returns 6 columns: [x1, y1, x2, y2, score, class_id]
        dets = np.asarray(kept_boxes, dtype=np.float32) if kept_boxes else np.zeros((0, 6), np.float32)
        lms = np.asarray(kept_lms, dtype=np.float32) if kept_lms else np.zeros((0, 10), np.float32)

        return dets, lms
