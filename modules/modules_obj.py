import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class ObjTracker:
    """Object detection for objects near the user's hand using YOLO."""

    def __init__(self, det_cooltime=10):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Assuming 'pretrained/object/yolo11m.pt' is available in the project structure
        YOLO_obj_path = os.path.join(repo_root, 'pretrained', 'object', 'yolo11m.pt')
        self.detector_obj = YOLO(YOLO_obj_path)
        self.detector_obj.to(self.device)

        # Initialize counter/flag for optional cooltime logic
        self.det_cooltime = det_cooltime
        self.obj_cnt = 0
        self.flag_detected = False

    def detect_objs_no_cnt(self, img, depth_image_float, d_wrist, depth_threshold=0.1):
        """
        Detects objects in the image and filters them based on their depth proximity to the wrist.
        This method runs detection on every call without cooltime logic.

        Args:
            img (np.ndarray): The RGB image frame.
            depth_image_float (np.ndarray): The PV-aligned depth image in meters.
            d_wrist (float): The depth of the wrist.
            depth_threshold (float): Maximum depth difference (in meters) from wrist to consider an object 'nearby'.

        Returns:
            list: A list of nearby object bounding boxes and labels [[x1, y1, x2, y2, label], ...].
        """
        self.flag_detected = True

        # Create a depth mask for areas close to the wrist depth
        mask = (depth_image_float > 0) & (np.abs(depth_image_float - d_wrist) <= depth_threshold)

        results = self.detector_obj(img, verbose=False)

        obj_bb_nearby = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = result.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == 'person':
                    continue

                # Check if the object's center is within the depth mask
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Check bounds before accessing mask
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx]:
                    obj_bb_nearby.append([x1, y1, x2, y2, label])

        return obj_bb_nearby

    # Keeping the original detect_objs for compatibility, though detect_objs_no_cnt is used in the main logic.
    def detect_objs(self, img, depth_image_float, d_wrist):
        """Detects objects with a cool-down period."""
        self.obj_cnt += 1

        if self.obj_cnt > self.det_cooltime:
            self.obj_cnt = 0
            return self.detect_objs_no_cnt(img, depth_image_float, d_wrist)
        else:
            self.flag_detected = False
            return []