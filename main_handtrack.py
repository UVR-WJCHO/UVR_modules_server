import sys, os
import time
from collections import deque
import cv2
import numpy as np
import struct
import socket
import keyboard

# Make packages under modules/ importable as top-level (handtracker, meshrecon, ...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

# Import refactored modules
from modules_hand import HandTracker_our, HandTracker_our_wilor
from modules_gesture import GestureClassfier
from modules_obj import ObjTracker
from modules_hl2 import Hl2Manager


# --- CONFIGURATION ---
CKPT_FILE = "checkpoint-40.tar"
FLAG_INTERACTION_DETECT = False  # Set to True to enable object-aware gesture detection

# HoloLens2 IP Address and UDP Port
HOST = '192.168.50.31'
UDP_PORT = 5005

# Front RGB camera parameters (Matching HL2SS configuration)
PV_WIDTH = 640
PV_HEIGHT = 360
PV_FPS = 30

# Depth image processing frequency
NUM_DEPTH_COUNT = 10  # Process depth once every N RGB frames

# Gesture sequence parameters
SEQ_LEN = 16
THRESHOLD_NUM = 5  # Number of consecutive frames required for a valid gesture


# --- HELPER FUNCTIONS (Placeholder for external utility) ---
def draw_2d_skeleton(image, joints_2d):
    """Draws a simple 2D skeleton on the image (placeholder)."""
    # Assuming joints_2d is (21, 2)
    for i, (u, v) in enumerate(joints_2d.astype(np.int32)):
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            cv2.circle(image, (u, v), 3, (255, 0, 0), -1)
    return image


# --- MAIN APPLICATION ---
def main():
    """Main execution loop for HL2 data processing and gesture/object recognition."""

    # 1. Initialize Models
    try:
        track_hand_v1 = HandTracker_our()
        track_hand_v2 = HandTracker_our_wilor()
        flag_hand_model = True  # Start with v2

        track_gesture = GestureClassfier(
            ckpt=f"./modules/gestureclassifier/checkpoints/{CKPT_FILE}",
            seq_len=SEQ_LEN,
            model_opt=1
        )

        track_obj = ObjTracker() if FLAG_INTERACTION_DETECT else None

    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    # 2. Initialize HoloLens2 Communication
    hl2_manager = Hl2Manager(HOST, PV_WIDTH, PV_HEIGHT, PV_FPS)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 3. Setup Visualization
    cv2.namedWindow('Prompt')
    cv2.resizeWindow(winname='Prompt', width=500, height=500)
    # cv2.moveWindow(winname='Prompt', x=2000, y=200) # Optional: move window off-screen

    # 4. State Variables
    idx_depth = 0
    flag_cooldown = False
    t_cooldown = 0.0
    queue_righthand = deque([], maxlen=SEQ_LEN)
    prev_gesture = None
    gesture_cnt = 0
    valid_gesture = None
    valid_gesture_idx = -1
    debug_pose = np.ones((21, 3))  # Dummy pose for non-detection

    try:
        while True:
            # Switch hand model on 'space' press (if keyboard access is available)
            current_hand_tracker = track_hand_v2 if flag_hand_model else track_hand_v1
            if keyboard.is_pressed('space'):
                flag_hand_model = not flag_hand_model
                print(f"Hand model switched to {'v2' if flag_hand_model else 'v1'}")

            # Determine if depth image should be processed this frame
            idx_depth = (idx_depth + 1) % (NUM_DEPTH_COUNT + 1)
            flag_depth = (NUM_DEPTH_COUNT > 0 and idx_depth == 0)

            # 5. Receive Input
            result = hl2_manager.receive_images(flag_depth)
            if result is None:
                continue
            color, depth, _ = result

            # Display RGBD pair
            cv2.imshow('RGB', color)
            if flag_depth and depth is not None:
                cv2.imshow('Depth', depth / hl2_manager.max_depth)
            cv2.waitKey(1)

            # Resize color image for consistent hand tracking input
            color_resized = cv2.resize(color, dsize=(PV_WIDTH, PV_HEIGHT), interpolation=cv2.INTER_AREA)

            # 6. Process Hand
            outs = current_hand_tracker.run(np.copy(color_resized))  # uvd_right
            if not isinstance(outs, np.ndarray):
                valid_gesture = None
                valid_gesture_idx = -1
                gesture_cnt = 0
                continue

            # 7. Process Gesture Sequence Data
            angle_label = track_gesture._compute_ang_from_joint(outs)
            data = np.concatenate([outs.flatten(), angle_label])
            queue_righthand.append(data)

            # 8. Interaction Prediction (Object-aware activation)
            flag_gesture = False
            if FLAG_INTERACTION_DETECT and flag_depth and depth is not None and track_obj is not None:
                # Get wrist depth and check object proximity
                uv_wrist = outs[0, :2].astype(int)
                if 0 <= uv_wrist[1] < PV_HEIGHT and 0 <= uv_wrist[0] < PV_WIDTH:
                    d_wrist = depth[uv_wrist[1], uv_wrist[0]]
                    obj_nearby_list = track_obj.detect_objs_no_cnt(color_resized, depth, d_wrist)

                    if len(obj_nearby_list) > 0:
                        # Calculate palm center and check 2D distance to nearby object centers
                        palm_points_2d = outs[[0, 4, 8, 12, 16, 20], :2]
                        palm_uv = np.mean(palm_points_2d, axis=0)

                        for obj_nearby in obj_nearby_list:
                            cx, cy = (obj_nearby[0] + obj_nearby[2]) // 2, (obj_nearby[1] + obj_nearby[3]) // 2
                            distance = np.sqrt((cx - palm_uv[0]) ** 2 + (cy - palm_uv[1]) ** 2)
                            if distance < 70.0:  # 2D pixel proximity threshold
                                flag_gesture = True
                                break
            else:
                # Always enable gesture recognition if interaction detection is off or depth is unavailable
                flag_gesture = True

            # 9. Run Gesture Classifier
            gesture = None
            if flag_gesture and len(queue_righthand) == SEQ_LEN:
                gesture_idx, gesture = track_gesture.run(queue_righthand)
            else:
                gesture_idx = -1

            # Valid gesture check (continuous detection)
            if prev_gesture == gesture and gesture != 'Natural' and gesture is not None:
                gesture_cnt += 1
            else:
                gesture_cnt = 0
            prev_gesture = gesture

            if gesture_cnt > THRESHOLD_NUM:
                valid_gesture = gesture
                valid_gesture_idx = gesture_idx
                gesture_cnt = 0  # Reset counter after a valid gesture is registered
            elif gesture_cnt == 0:
                valid_gesture = None
                valid_gesture_idx = -1

            # 10. Visualization
            vis_color = draw_2d_skeleton(color_resized.copy(), outs[:, :2])

            if valid_gesture is not None and valid_gesture != "Natural":
                cv2.putText(vis_color, f'{valid_gesture.upper()}',
                            org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255),
                            thickness=3)

            cv2.imshow("Prompt", vis_color)
            cv2.waitKey(1)

            # 11. Send to HoloLens2
            # Cooldown logic (0.5 second delay between sending valid gestures)
            if time.time() - t_cooldown > 0.5:
                flag_cooldown = False

            if not flag_cooldown and valid_gesture is not None and valid_gesture != "Natural":
                # Send valid gesture data: flattened pose (63) + gesture index (1) + timestamp (1)
                send_data = outs.flatten().tolist() + [float(valid_gesture_idx), float(time.time() * 1000)]
                flag_cooldown = True
                t_cooldown = time.time()
                print(f"Sending gesture: {valid_gesture}")
            else:
                # Send non-detection or debug data: dummy pose (63) + invalid index (-1) + timestamp (1)
                send_data = debug_pose.flatten().tolist() + [float(-1), float(time.time() * 1000)]

            # UDP packet sending (format: 65 doubles)
            fmt = f"{len(send_data)}d"
            send_bytes = struct.pack(fmt, *send_data)
            sock.sendto(send_bytes, (HOST, UDP_PORT))

    except Exception as e:
        print(f"An error occurred in the main loop: {e}")

    finally:
        print("Cleaning up resources...")
        sock.close()
        hl2_manager.destroy()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()