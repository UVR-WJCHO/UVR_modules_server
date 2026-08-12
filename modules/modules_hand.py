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

class HandTracker_onnx():
    # ONNX-based WILOR hand tracker. onnxruntime is imported lazily inside
    # __init__ so importing this module stays cheap for users of the other
    # trackers (mediapipe).
    def __init__(self, **kwargs):
        from handtracker_onnx import WilorHandTrackerONNX
        self.model_hand = WilorHandTrackerONNX(**kwargs)

    def warmup(self, image=None):
        self.model_hand.warmup(image)

    def run(self, input):
        # input : img_cv (BGR). Returns (21, 3) [u, v, z] to match the other
        # trackers' output contract, or None when no hand is detected.
        result = self.model_hand.process(input)
        if result is None:
            return None
        joints_2d = result['joints_2d']       # (21, 2) pixel coords
        # joints_3d 는 metre. 다른 tracker 들은 z 를 mm 로 돌려주므로 맞춰준다.
        z = result['joints_3d'][:, 2:3] * 1000.0   # (21, 1) wrist-relative depth, mm
        return np.concatenate([joints_2d, z], axis=1)


class HandTracker_mp():
    def __init__(self, ckpt=None):

        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        print("init hand tracker")
        torch.backends.cudnn.benchmark = True
        self.mediahand = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

    def run(self, input):
        img_height = input.shape[0]
        img_width = input.shape[1]

        input = cv2.flip(input, 1)
        results = self.mediahand.process(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))

        result_hand = []
        if results.multi_hand_landmarks == None:
            return None

        for hand_landmarks in results.multi_hand_landmarks:
            for _, landmark in enumerate(hand_landmarks.landmark):
                x = img_width - int(landmark.x * img_width)
                y = int(landmark.y * img_height)
                z = landmark.z
                result_hand.append([x, y, z])
        result_hand = np.asarray(result_hand)

        return result_hand
