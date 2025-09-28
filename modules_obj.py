import os
import sys
import cv2
import time
import numpy as np
from ultralytics import YOLO

import torch
import mediapipe as mp
from collections import deque
from enum import Enum, IntEnum
import copy



class ObjTracker():
    def __init__(self, det_cooltime=10):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        YOLO_obj_path = os.path.join(curr_dir, 'pretrain', 'yolo11m.pt')
        self.detector_obj = YOLO(YOLO_obj_path)

        self.detector_obj.to(self.device)

        testImg = cv2.imread(os.path.join(curr_dir, './examples/test1.jpg'))
        testImg = cv2.resize(testImg, (640, 360))
        _ = self.detector_obj(testImg, verbose=False)

        self.det_cooltime = det_cooltime
        self.obj_cnt = 0
        self.flag_detected = False

    def detect_objs(self, img, depth_image_float, d_wrist):
        self.obj_cnt += 1

        ## run YOLO when every cooltime
        if self.obj_cnt > self.det_cooltime:
            self.flag_detected = True
            self.obj_cnt = 0

            mask = (depth_image_float > 0) & (depth_image_float - d_wrist <= 0.1)
            mask = mask.astype(np.uint8) * 255
            # masked_rgb = cv2.bitwise_and(img, img, mask=mask)

            # 절반 사이즈로 YOLO 돌린후 결과*2
            # resized_img = cv2.resize(img, (self.img_w // 2, self.img_h // 2), interpolation=cv2.INTER_AREA)
            results = self.detector_obj(img, verbose=False)

            # debug_vis = img.copy()
            obj_bb_nearby = []
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = result.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # cv2.rectangle(debug_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # cv2.putText(debug_vis, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if label == 'person':
                        continue

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if mask[int(cy), int(cx)] == False:
                        continue

                    # x1, y1, x2, y2 = 2 * x1, 2 * y1, 2 * x2, 2 * y2
                    obj_bb_nearby.append([x1, y1, x2, y2, label])

            # cv2.imshow("debug", debug_vis)

            return obj_bb_nearby
        else:
            self.flag_detected = False
            return []


    def detect_objs_no_cnt(self, img, depth_image_float, d_wrist):
        self.flag_detected = True

        mask = (depth_image_float > 0) & (depth_image_float - d_wrist <= 0.1)
        mask = mask.astype(np.uint8) * 255
        # masked_rgb = cv2.bitwise_and(img, img, mask=mask)

        # 절반 사이즈로 YOLO 돌린후 결과*2
        # resized_img = cv2.resize(img, (self.img_w // 2, self.img_h // 2), interpolation=cv2.INTER_AREA)
        results = self.detector_obj(img, verbose=False)

        # debug_vis = img.copy()
        obj_bb_nearby = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = result.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # cv2.rectangle(debug_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.putText(debug_vis, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if label == 'person':
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if mask[int(cy), int(cx)] == False:
                    continue

                # x1, y1, x2, y2 = 2 * x1, 2 * y1, 2 * x2, 2 * y2
                obj_bb_nearby.append([x1, y1, x2, y2, label])

        # cv2.imshow("debug", debug_vis)

        return obj_bb_nearby



class ObjTracker_old():
    def __init__(self):
        self.model = YOLO("./objecttracker/yolo11n.yaml")
        self.model = YOLO("./objecttracker/yolo11n.pt").to('cuda')
        self.idx = 0

    def run(self, img, flag_vis=False): # input : img_cv
        # imgSize = (img.shape[0], img.shape[1])  # (360, 640)

        # results = self.model(img, conf=0.4, device=0)
        results = self.model(img, conf=0.4, device=0, verbose=False, classes=[0, 39, 41, 43, 44, 46, 47, 64, 65, 67])
        result = results[0]

        if flag_vis:
            plots = result.plot()
            cv2.imshow("object tracker results", plots)
            cv2.waitKey(1)

        boxes = result.boxes

        center_dict = {}
        flag_hand = False
        for box in boxes:
            bbox = np.squeeze(box.xyxy.cpu().numpy())
            cls = int(box.cls.cpu().numpy()[0])
            if cls == 0:
                # hand detected
                flag_hand = True
            if cls != 0:
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                center_dict[cls] = [center_x, center_y]

        ## visualize obj centers
        # if flag_vis:
            # debug = np.copy(img)
            # for center in center_list:
            #     cv2.circle(debug, center, 5, color=[255, 255, 0], thickness=-1, lineType=cv2.LINE_AA)
            # cv2.imshow("object centers", debug)
            # cv2.waitKey(1)

        return flag_hand, center_dict


# 0: 'person', 41: 'cup',
"""
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 
33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
  49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
   58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
   66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
"""

