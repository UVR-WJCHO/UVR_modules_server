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

from handtracker.module_SARTE import HandTracker
from handtracker_wilor.module_WILOR import HandTracker_wilor


class HandTracker_our_wilor():
    def __init__(self):
        self.model_hand = HandTracker_wilor()

    def run(self, input):
        return self.model_hand.run(input)



class HandTracker_our():
    def __init__(self):
        self.track_hand = HandTracker()

    def run(self, input):
        result_hand = self.track_hand.Process_single_newroi(input)

        return result_hand


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
