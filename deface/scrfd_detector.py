import os
import cv2
import numpy as np 


def ensure_rgb(self, img):

        """Convert input image to RGB if it is in RGBA or L format"""
        if img.ndim == 2:  # 1-channel grayscale -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # 4-channel RGBA -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img


class SCRFDdetector:
    def __init__(self, model_path: str, device: str = "cpu", cap_long_side: int = 1920, override_execution_provider: str = None):
        import onnxruntime
        from .scrfd import SCRFD

        self.cap_long_side = cap_long_side  # Maximum long side length for input frames

        providers = onnxruntime.get_available_providers()   # Get list of available providers & choose one
         # If no override, use all available providers
        if override_execution_provider is None:
            ort_providers = providers
        else:
            if override_execution_provider in providers:
                ort_providers = [override_execution_provider]
            else:
                raise ValueError(f"Requested execution provider '{override_execution_provider}' is not available. Available providers: {providers}")
        self.model = SCRFD(model_path, device=device, providers=ort_providers)

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        #Resize input frame to have its long side equal to cap_long_side while maintaining aspect ratio
        h, w = frame.shape[:2]
        if max(h, w) > self.cap_long_side:
            scale = self.cap_long_side / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h)) 
        #returning resized frame else original frame
        return frame
    
    def rgb_to_bgr(self, img: np.ndarray) -> np.ndarray:
        """Convert input image from RGB to BGR format"""
        img_bgr=cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32)
        img_bgr = (img_bgr -127.5) / 128.0
        return img_bgr
    
    def infer(self, frame: np.ndarray):
        frame_resized = self.resize_frame(frame)
        #img_bgr = self.rgb_to_bgr(frame_resized)
        bboxes, kpss = self.model.detect(frame_resized, max_num=0, metric='default', iou_thresh=0.5, score_thresh=0.5)
        return bboxes, kpss