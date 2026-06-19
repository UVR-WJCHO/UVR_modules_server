import os, sys
import time
import cv2
import numpy as np
from PIL import Image
import socket
import torch
import select, tty, termios

from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread
import gc

from modules_mesh import MeshReconstructor
from modules_segment import HOSegmentor
from modules_hotrack import InteractiveHoTrackSegmentor
from modules_hl2 import Hl2Manager


## check before execution
# - sudo lsof -i :8000
# - sudo kill -9 {}


## Set HoloLens2 options ##
host = '192.168.50.137'  # HoloLens2 wifi address

pv_width = 1280
pv_height = 720
pv_fps = 30

fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

# HTTP 서버 설정
HTTP_PORT = 8000

flag_skip_mesh = False
flag_interactive_hotrack = os.getenv("UVR_USE_INTERACTIVE_HOTRACK", "1") != "0"


class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """GLB 파일을 제공하는 커스텀 HTTP 핸들러"""

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[HTTP] {format % args}")


def start_http_server():
    """백그라운드에서 HTTP 서버 실행"""
    server = HTTPServer(('0.0.0.0', HTTP_PORT), CustomHTTPRequestHandler)
    print(f"[HTTP Server] Started on port {HTTP_PORT}")
    server.serve_forever()


def clear_gpu_memory():
    """GPU 메모리 완전히 비우기"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("[GPU] Memory cleared")


def print_gpu_memory():
    """현재 GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"[GPU] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def main():
    clear_gpu_memory()

    ###################### HTTP 서버 시작 ######################
    http_thread = Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(0.5)

    ###################### init models ######################
    if flag_interactive_hotrack:
        print("\n[Init] Initializing InteractiveHoTrackSegmentor...")
        hosegmentor = InteractiveHoTrackSegmentor(
            output_dir=os.getenv("UVR_HOTRACK_OUTPUT_DIR", "output/hotrack_stage1"),
            video_name=os.getenv("UVR_HOTRACK_VIDEO_NAME", "hl2_online"),
            yolo_model_path=os.getenv("UVR_HOTRACK_YOLO_MODEL", "segmentor/100DOH_small.pt"),
            sam2_variant=os.getenv("UVR_HOTRACK_SAM2_VARIANT", "tiny"),
            sam2_checkpoint=os.getenv("UVR_HOTRACK_SAM2_CHECKPOINT", ""),
            max_side=int(os.getenv("UVR_HOTRACK_MAX_SIDE", "960")),
            detect_interval=int(os.getenv("UVR_HOTRACK_DETECT_INTERVAL", "3")),
            max_active_objects=int(os.getenv("UVR_HOTRACK_MAX_OBJECTS", "5")),
            max_active_hands=int(os.getenv("UVR_HOTRACK_MAX_HANDS", "2")),
            track_hands=os.getenv("UVR_HOTRACK_TRACK_HANDS", "0") == "1",
            include_hands=os.getenv("UVR_HOTRACK_INCLUDE_HANDS", "0") == "1",
            hand_backend=os.getenv("UVR_HOTRACK_HAND_BACKEND", "auto"),
            save_masks=os.getenv("UVR_HOTRACK_SAVE_MASKS", "1") != "0",
            interactive_window=os.getenv("UVR_HOTRACK_WINDOW", "1") != "0",
            offload_video_to_cpu=os.getenv("UVR_HOTRACK_OFFLOAD_VIDEO", "1") != "0",
            offload_state_to_cpu=os.getenv("UVR_HOTRACK_OFFLOAD_STATE", "1") != "0",
            state_window=int(os.getenv("UVR_HOTRACK_STATE_WINDOW", "9")),
            log_memory=os.getenv("UVR_HOTRACK_LOG_MEMORY", "0") == "1",
        )
    else:
        print("\n[Init] Initializing legacy HOSegmentor...")
        hosegmentor = HOSegmentor()
    print("\n[Init] Initializing MeshReconstructor...")
    if not flag_skip_mesh:
        meshrecon = MeshReconstructor()

    ###################### init hl2 ######################
    print("\n[Init] Initializing Hl2Manager...")
    hl2_manager = Hl2Manager(host, pv_width, pv_height, pv_fps)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)

    is_first_call = True
    frame_idx = 0
    obj_idx = 0

    print("\n[Init] Starting loop")
    try:
        while True:
            ###################### receive input ######################
            result = hl2_manager.receive_images(flag_depth=True)
            if result == None:
                print("no results")
                continue

            color, depth, intrinsic_per_frame = result

            cv2.imshow('RGB', color)
            rgb_key = cv2.waitKey(1)
            if flag_interactive_hotrack:
                hosegmentor.handle_key(rgb_key)
                if hosegmentor.quit_requested:
                    break

            ###################### process ######################
            if flag_interactive_hotrack:
                with torch.inference_mode():
                    h_color_image, result_masks, _ = hosegmentor.process_frame(color.copy())
                    key = cv2.waitKey(1)
                    hosegmentor.handle_key(key)
                    frame_idx += 1
                    if hosegmentor.quit_requested:
                        break
                    if len(result_masks) == 0:
                        continue
                    obj_mask = result_masks[0]
                    masked_color = np.where(obj_mask[..., None] == 1, color, 3)
                    cv2.imshow("Segmentation", masked_color)
                    key = cv2.waitKey(1)
                    hosegmentor.handle_key(key)
                    if hosegmentor.quit_requested:
                        break
            else:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    depth_cm = depth * 10.0
                    hotrack_datas = hosegmentor.get_hotrack_datas(color, depth_cm)

                    h_color_image, result_masks = hosegmentor.run(color.copy(), depth_cm, hotrack_datas,
                                                                  is_first_call,
                                                                  frame_idx)
                    is_first_call = False
                    frame_idx += 1

                    if len(result_masks) == 0:
                        continue

                    obj_mask = result_masks[0]

                    masked_color = np.where(obj_mask[..., None] == 1, color, 3)
                    cv2.imshow("Segmentation", masked_color)
                    cv2.waitKey(1)

            # 스페이스바 입력 확인
            dr, dw, de = select.select([sys.stdin], [], [], 0.01)
            if dr:
                key = sys.stdin.read(1)
                if key == ' ':
                    print("\n[Mesh] Starting reconstruction...")

                    masked_color_pil = Image.fromarray(masked_color)

                    cv2.imwrite((f"./rgb_{obj_idx}.png"), color)
                    cv2.imwrite((f"./rgb_masked_{obj_idx}.png"), masked_color)
                    np.save(f"./depth_{obj_idx}.npy", depth)
                    print("save images")

                    # save auto adjusted intrinsic per frame
                    np.save(f"./intrinsic_{obj_idx}.npy", intrinsic_per_frame)
                    print("intrinsic : ", intrinsic_per_frame)

                    if not flag_skip_mesh:
                        try:
                            mesh_glb = meshrecon.run(masked_color_pil)

                            output_path = f"./mesh_{obj_idx}.glb"
                            mesh_glb.export(output_path)
                            print(f"[Mesh] Saved to {output_path}")

                            # HoloLens2에 신호 전송
                            signal = b"1"
                            sock.sendto(signal, (host, 5005))
                            print(f"[UDP] Signal sent to {host}:5005")

                            # local_ip = sock_check.gethostbyname(sock_check.gethostname())
                            # print(f"[Info] GLB available at http://{local_ip}:{HTTP_PORT}/output.glb")

                        except Exception as e:
                            print(f"[Error] Mesh reconstruction failed: {e}")
                        finally:
                            # 메시 재구성 객체 삭제 및 메모리 정리
                            del mesh_glb
                            del masked_color_pil

                            clear_gpu_memory()
                            print_gpu_memory()

                    obj_idx += 1

                    print("[Info] Ready for next reconstruction (Press Space)")
                    # break 제거 - 계속 루프 실행

    except KeyboardInterrupt:
        print("\n[Info] Shutting down...")

    finally:
        if not flag_skip_mesh:
            del meshrecon
        if hasattr(hosegmentor, "close"):
            hosegmentor.close()
        del hosegmentor
        clear_gpu_memory()

        sock.close()
        hl2_manager.destroy()
        cv2.destroyAllWindows()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()