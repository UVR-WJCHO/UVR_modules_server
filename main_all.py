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
from modules_hl2 import Hl2Manager


## check before execution
# - sudo lsof -i :8000
# - sudo kill -9 {}


## Set HoloLens2 options ##
host = '192.168.50.31'  # HoloLens2 wifi address

pv_width = 1280
pv_height = 720
pv_fps = 30

fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

# HTTP 서버 설정
HTTP_PORT = 8000


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
    hosegmentor = HOSegmentor()
    meshrecon = MeshReconstructor()

    ###################### init hl2 ######################
    hl2_manager = Hl2Manager(host, pv_width, pv_height, pv_fps)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)

    is_first_call = True
    frame_idx = 0

    try:
        while True:
            ###################### receive input ######################
            result = hl2_manager.receive_images(flag_depth=True)
            if result == None:
                continue

            color, depth = result

            cv2.imshow('RGB', color)
            cv2.waitKey(1)

            ###################### process ######################
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

                    try:
                        mesh_glb = meshrecon.run(masked_color_pil)

                        output_path = "./output.glb"
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

                    print("[Info] Ready for next reconstruction (Press Space)")
                    # break 제거 - 계속 루프 실행

    except KeyboardInterrupt:
        print("\n[Info] Shutting down...")

    finally:
        del meshrecon
        del hosegmentor
        clear_gpu_memory()

        sock.close()
        hl2_manager.destroy()
        cv2.destroyAllWindows()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()