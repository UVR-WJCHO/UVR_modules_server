import os, sys

# Make packages under modules/ importable as top-level (meshrecon, segmentor, hotrack, ...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

import datetime
import cv2
import numpy as np

from modules_console import enable_cbreak_stdin, read_key_nonblocking, restore_stdin_cbreak
from modules_hl2 import Hl2Manager


## Set HoloLens2 options ##
host = os.environ.get('UVR_HL2_HOST', '192.168.50.137')  # HoloLens2 wifi address

pv_width = 1280
pv_height = 720
pv_fps = 30


def start_take():
    """녹화 시작 시점의 현재 시간으로 output 하위 폴더와 하위 구조를 만든다."""
    take_dir = os.path.join("output", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    for sub in ("color", "depth", "intrinsics"):
        os.makedirs(os.path.join(take_dir, sub), exist_ok=True)
    print(f"\n[Record] Started -> {take_dir}")
    return take_dir


def save_frame(take_dir, frame_idx, color, depth, intrinsic_per_frame):
    name = f"{frame_idx:06d}"
    cv2.imwrite(os.path.join(take_dir, "color", f"{name}.png"), color)
    np.save(os.path.join(take_dir, "depth", f"{name}.npy"), depth)
    np.save(os.path.join(take_dir, "intrinsics", f"{name}.npy"), intrinsic_per_frame)


def main():
    os.makedirs("output", exist_ok=True)

    ###################### init hl2 ######################
    print("\n[Init] Initializing Hl2Manager...")
    hl2_manager = Hl2Manager(host, pv_width, pv_height, pv_fps)

    recording = False
    take_dir = None
    frame_idx = 0

    print("\n[Init] Starting loop (Space: start/stop recording, q or ESC: quit)")
    console_state = enable_cbreak_stdin()
    try:
        while True:
            ###################### receive input ######################
            result = hl2_manager.receive_images(flag_depth=True)
            if result == None:
                print("no results")
                continue

            color, depth, intrinsic_per_frame = result

            ###################### record ######################
            if recording:
                save_frame(take_dir, frame_idx, color, depth, intrinsic_per_frame)
                frame_idx += 1

            preview = color.copy()
            if recording:
                cv2.putText(preview, f"REC {frame_idx}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow('RGB', preview)

            ###################### key handling ######################
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                key = read_key_nonblocking(timeout=0.01)
                key = ord(key) if key else 255

            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                if recording:
                    print(f"[Record] Stopped: {frame_idx} frames saved to {take_dir}")
                    recording = False
                    take_dir = None
                else:
                    take_dir = start_take()
                    frame_idx = 0
                    recording = True

    except KeyboardInterrupt:
        print("\n[Info] Shutting down...")

    finally:
        if recording:
            print(f"[Record] Stopped: {frame_idx} frames saved to {take_dir}")
        hl2_manager.destroy()
        cv2.destroyAllWindows()
        restore_stdin_cbreak(console_state)


if __name__ == '__main__':
    main()
