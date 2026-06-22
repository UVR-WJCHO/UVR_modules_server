import os, sys

# Make packages under modules/ importable as top-level (meshrecon, segmentor, hotrack, ...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

import time
import datetime
import cv2
import numpy as np
from PIL import Image
import torch
import select, tty, termios

from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread
import gc

from modules_mesh import MeshReconstructor
from modules_hotrack import InteractiveHoTrackSegmentor
from modules_behavior import BehaviorPropertyEstimator


## Webcam options ##
WEBCAM_INDEX = 0          # /dev/videoN index
cam_width = 1280
cam_height = 720

fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

# HTTP 서버 설정
HTTP_PORT = 8000

flag_recon_mesh = True
flag_behavior = False  # run behavior property estimation (GLB -> property JSON) after each mesh


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
    os.makedirs("output", exist_ok=True)

    ###################### HTTP 서버 시작 ######################
    http_thread = Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(0.5)

    ###################### init models ######################
    # 웹캠은 color만 제공하므로 depth 기반 legacy HOSegmentor 대신 color-only HoTrack 경로만 사용
    print("\n[Init] Initializing InteractiveHoTrackSegmentor...")
    hosegmentor = InteractiveHoTrackSegmentor(output_dir="output/hotrack_stage1")
    print("\n[Init] Initializing MeshReconstructor...")
    if flag_recon_mesh:
        meshrecon = MeshReconstructor()
    if flag_behavior:
        print("\n[Init] Initializing BehaviorPropertyEstimator...")
        behavior_estimator = BehaviorPropertyEstimator()

    ###################### init webcam ######################
    print("\n[Init] Opening webcam...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam (index {WEBCAM_INDEX})")

    frame_idx = 0

    print("\n[Init] Starting loop")
    try:
        while True:
            ###################### receive input ######################
            ret, frame = cap.read()
            if not ret:
                print("no frame")
                continue

            # OpenCV는 BGR로 읽으므로 HL2(RGB) 파이프라인과 동일하게 RGB로 변환
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imshow('RGB', color)
            rgb_key = cv2.waitKey(1)
            hosegmentor.handle_key(rgb_key)
            if hosegmentor.quit_requested:
                break

            ###################### process ######################
            with torch.inference_mode():
                h_color_image, result_masks, _ = hosegmentor.process_frame(color.copy())
                hosegmentor.handle_key(cv2.waitKey(1))
                if hosegmentor.quit_requested:
                    break
            frame_idx += 1

            if len(result_masks) == 0:
                continue

            obj_mask = result_masks[0]
            masked_color = np.where(obj_mask[..., None] == 1, color, 3)
            cv2.imshow("Segmentation", masked_color)
            key = cv2.waitKey(1)
            hosegmentor.handle_key(key)
            if hosegmentor.quit_requested:
                break

            # 스페이스바 입력 확인
            dr, dw, de = select.select([sys.stdin], [], [], 0.01)
            if dr:
                key = sys.stdin.read(1)
                if key == ' ':
                    print("\n[Mesh] Starting reconstruction...")

                    # 캡처마다 현재 시간으로 하위 폴더를 만들어 모든 결과물을 그 안에 저장
                    capture_dir = os.path.join("output", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                    os.makedirs(capture_dir, exist_ok=True)

                    masked_color_pil = Image.fromarray(masked_color)

                    cv2.imwrite(os.path.join(capture_dir, "rgb.png"), color)
                    cv2.imwrite(os.path.join(capture_dir, "rgb_masked.png"), masked_color)
                    print("save images")

                    if flag_recon_mesh:
                        try:
                            mesh_glb = meshrecon.run(masked_color_pil)

                            output_path = os.path.join(capture_dir, "mesh.glb")
                            mesh_glb.export(output_path)
                            print(f"[Mesh] Saved to {output_path}")

                            if flag_behavior:
                                try:
                                    print("\n[Behavior] Estimating properties from GLB...")
                                    json_path = behavior_estimator.run(
                                        output_path,
                                        output_json=os.path.join(capture_dir, "property.json"),
                                        vlm_input_dir=capture_dir,
                                    )
                                    print(f"[Behavior] Property JSON saved to {json_path}")
                                except Exception as e:
                                    print(f"[Error] Behavior estimation failed: {e}")

                        except Exception as e:
                            print(f"[Error] Mesh reconstruction failed: {e}")
                        finally:
                            # 메시 재구성 객체 삭제 및 메모리 정리
                            del mesh_glb
                            del masked_color_pil

                            clear_gpu_memory()
                            print_gpu_memory()

                    print("[Info] Ready for next reconstruction (Press Space)")

    except KeyboardInterrupt:
        print("\n[Info] Shutting down...")

    finally:
        if flag_recon_mesh:
            del meshrecon
        if hasattr(hosegmentor, "close"):
            hosegmentor.close()
        del hosegmentor
        clear_gpu_memory()

        cap.release()
        cv2.destroyAllWindows()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
