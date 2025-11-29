import os, sys
import asyncio
import time
from collections import deque
import cv2
import numpy as np
import websockets
import struct
import json
from PIL import Image, ImageDraw, ImageFont


import socket, threading
import multiprocessing as mp

# from modules_hand import HandTracker_mp, HandTracker_our
# from utils.visualize import draw_2d_skeleton

# from modules_gesture import GestureClassfier, recog_contact
# from modules_obj import ObjTracker

from modules_mesh import MeshReconstructor
from modules_hl2 import Hl2Manager


## Set HoloLens2 options ##
host = '192.168.50.31'  # HoloLens2 wifi address

pv_width = 1280
pv_height = 720
pv_fps = 30

num_depth_count = 1

def main():
    ###################### init models ######################

    # receiver
    hl2_manager = Hl2Manager(host, pv_width, pv_height, pv_fps)
    # transmitter
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    idx_depth = 0
    while True:
        ###################### receive input ######################
        idx_depth += 1
        if idx_depth == num_depth_count:
            idx_depth = 0
            flag_depth = True
        else:
            flag_depth = False

        result = hl2_manager.receive_images(flag_depth)
        if result == None:
            continue

        color, depth = result

        ## Display RGBD pair
        # cv2.imshow('RGB', color)
        # if flag_depth:
        #     cv2.imshow('Depth', depth / max_depth)  # scale for visibility
        # cv2.waitKey(1)

        ###################### process ######################


        output = True

        ###################### send to hl2 ######################
        if output:
            send_data = outs.flatten().tolist() + [float(valid_gesture_idx), float(time.time() * 1000)]
        else:
            debug_pose = np.ones((63))
            send_data = debug_pose.tolist() + [float(-1), float(time.time() * 1000)]

        fmt = f"{len(send_data)}d"
        send_bytes = struct.pack(fmt, *send_data)
        sock.sendto(send_bytes, (host, 5005))

    sock.close()

    hl2_manager.destory()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
