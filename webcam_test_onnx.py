"""
Standalone webcam test for the ONNX WILOR hand tracker
(modules/handtracker_onnx).

Exercises WilorHandTrackerONNX end-to-end and overlays the 2D skeleton, hand
bbox, FPS and inference timing.

Usage:
    # webcam (default camera 0)
    python webcam_test_onnx.py

    # single-image headless smoke test (no camera / GUI required)
    python webcam_test_onnx.py --image path/to/hand.jpg --save out.jpg --no-show
"""
import os
import sys
import argparse
import time

import cv2
import numpy as np

# Make packages under modules/ importable as top-level (matches main_handtrack.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

from handtracker_onnx import (
    WilorHandTrackerONNX,
    DEFAULT_ONNX_PATH,
    DEFAULT_CFG_PATH,
    DEFAULT_MANO_PATH,
    DEFAULT_DETECTOR_PATH,
)
from handtracker_onnx.wilor_onnx_utils import draw_2d_skeleton


def render(frame, result):
    output_img = draw_2d_skeleton(frame, result['joints_2d'])
    x1, y1, x2, y2 = result['bbox'].astype(int)
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(output_img, f"hand {result['is_right']}", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return output_img


def run_image(tracker, image_path, save_path, show):
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f'Could not read image: {image_path}')

    t0 = time.perf_counter()
    result = tracker.process(frame)
    dt = time.perf_counter() - t0

    print(f'providers: {tracker.providers}')
    print(f'inference: {dt * 1000:.1f} ms')
    if result is None:
        print('No hand detected.')
        return

    print("joints_2d shape:", result['joints_2d'].shape,
          "joints_3d shape:", result['joints_3d'].shape,
          "vertices shape:", result['vertices'].shape,
          "is_right:", result['is_right'])
    output_img = render(frame, result)

    if save_path:
        cv2.imwrite(save_path, output_img)
        print(f'Saved visualization to: {save_path}')
    if show:
        cv2.imshow('WILOR ONNX image test', output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam(tracker, cam_index):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam.')

    cv2.namedWindow('WILOR ONNX Webcam', cv2.WINDOW_NORMAL)
    print(f'providers: {tracker.providers}  (press ESC to quit)')

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        result = tracker.process(frame)
        t1 = time.perf_counter()

        if result is None:
            cv2.putText(frame, 'No hand detected', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow('WILOR ONNX Webcam', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        output_img = render(frame, result)
        fps = 1.0 / (t1 - frame_start) if t1 > frame_start else 0.0
        cv2.putText(output_img, f'FPS: {fps:.1f}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(output_img, f'infer: {(t1 - t0) * 1000:.1f} ms', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow('WILOR ONNX Webcam', output_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='Test the ONNX WILOR hand tracker')
    parser.add_argument('--onnx', type=str, default=DEFAULT_ONNX_PATH)
    parser.add_argument('--cfg', type=str, default=DEFAULT_CFG_PATH)
    parser.add_argument('--mano', type=str, default=DEFAULT_MANO_PATH)
    parser.add_argument('--detector', type=str, default=DEFAULT_DETECTOR_PATH)
    parser.add_argument('--camera', type=int, default=0, help='Webcam index')
    parser.add_argument('--image', type=str, default=None,
                        help='Run on a single image instead of the webcam')
    parser.add_argument('--save', type=str, default=None, help='Path to save the visualization')
    parser.add_argument('--no-show', dest='show', action='store_false',
                        help='Do not open a GUI window (headless)')
    return parser.parse_args()


def main():
    args = parse_args()
    tracker = WilorHandTrackerONNX(
        onnx_path=args.onnx, cfg_path=args.cfg,
        mano_path=args.mano, detector_path=args.detector,
    )
    if args.image:
        run_image(tracker, args.image, args.save, args.show)
    else:
        run_webcam(tracker, args.camera)


if __name__ == '__main__':
    main()
