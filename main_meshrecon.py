import os, sys

# Make packages under modules/ importable as top-level (meshrecon, segmentor, hotrack, ...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

import time
import datetime
import cv2
import numpy as np
from PIL import Image
import socket
import torch

from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread
import gc

from modules_console import enable_cbreak_stdin, read_key_nonblocking, restore_stdin_cbreak
from modules_mesh import MeshReconstructor
from modules_segment import HOSegmentor
from modules_hotrack import build_interactive_hotrack_segmentor_from_env
from modules_hl2 import Hl2Manager
from modules_behavior import BehaviorPropertyEstimator
from modules_texpaint import TexPaintRunner


## check before execution
# sudo lsof -i :8000
# sudo kill -9 {}


## Set HoloLens2 options ##
host = os.environ.get('UVR_HL2_HOST', '192.168.50.137')  # HoloLens2 wifi address

pv_width = 1280
pv_height = 720
pv_fps = 30

# HTTP 서버 설정
HTTP_PORT = 8000

flag_recon_mesh = True
flag_interactive_hotrack = True  # True: InteractiveHoTrackSegmentor, False: legacy HOSegmentor
flag_behavior = False  # run behavior property estimation (GLB -> property JSON) after each mesh
# When True, texture the TRELLIS geometry with Hunyuan3D-Paint (separate env, PBR)
# instead of using TRELLIS's own gaussian-baked texture. Requires flag_recon_mesh.
flag_texpaint = False 

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
    if flag_interactive_hotrack:
        print("\n[Init] Initializing InteractiveHoTrackSegmentor...")
        hosegmentor = build_interactive_hotrack_segmentor_from_env()
    else:
        print("\n[Init] Initializing legacy HOSegmentor...")
        hosegmentor = HOSegmentor()
    if flag_recon_mesh:
        print("\n[Init] Initializing MeshReconstructor...")
        meshrecon = MeshReconstructor()
    if flag_behavior:
        print("\n[Init] Initializing BehaviorPropertyEstimator...")
        behavior_estimator = BehaviorPropertyEstimator()
    if flag_texpaint:
        print("\n[Init] Initializing TexPaintRunner...")
        texpaint_runner = TexPaintRunner()

    ###################### init hl2 ######################
    print("\n[Init] Initializing Hl2Manager...")
    hl2_manager = Hl2Manager(host, pv_width, pv_height, pv_fps)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)

    is_first_call = True
    frame_idx = 0
    obj_idx = 0

    print("\n[Init] Starting loop")
    console_state = enable_cbreak_stdin()
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
                    hosegmentor.handle_key(cv2.waitKey(1))
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
            key = cv2.waitKey(1)
            if flag_interactive_hotrack:
                hosegmentor.handle_key(key)
                if hosegmentor.quit_requested:
                    break

            # 스페이스바 입력 확인
            key = read_key_nonblocking(timeout=0.01)
            if key == ' ':
                    print("\n[Mesh] Starting reconstruction...")

                    # 캡처마다 현재 시간으로 하위 폴더를 만들어 모든 결과물을 그 안에 저장
                    capture_dir = os.path.join("output", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                    os.makedirs(capture_dir, exist_ok=True)

                    # mask를 alpha로 전달 -> TRELLIS가 rembg 재분리를 건너뛰고 이 mask를
                    # 그대로 사용(배경이 mesh 판(slab)으로 붙는 artifact 방지)
                    masked_rgba = np.dstack([color, (obj_mask.astype(np.uint8) * 255)])
                    masked_color_pil = Image.fromarray(masked_rgba, "RGBA")

                    cv2.imwrite(os.path.join(capture_dir, "rgb.png"), color)
                    cv2.imwrite(os.path.join(capture_dir, "rgb_masked.png"), masked_rgba)
                    np.save(os.path.join(capture_dir, "depth.npy"), depth)
                    print("save images")

                    # save auto adjusted intrinsic per frame
                    np.save(os.path.join(capture_dir, "intrinsic.npy"), intrinsic_per_frame)
                    print("intrinsic : ", intrinsic_per_frame)

                    if flag_recon_mesh:
                        try:
                            mesh_glb, mesh_preview = meshrecon.run(masked_color_pil, return_preview=True)

                            # 생성 확인용 프리뷰(2뷰) 표시 + 저장
                            preview_bgr = cv2.cvtColor(mesh_preview, cv2.COLOR_RGB2BGR)
                            cv2.imshow("Mesh preview", preview_bgr)
                            cv2.waitKey(1)
                            cv2.imwrite(os.path.join(capture_dir, "mesh_preview.png"), preview_bgr)

                            output_path = os.path.join(capture_dir, "mesh.glb")
                            mesh_glb.export(output_path)
                            print(f"[Mesh] Saved to {output_path}")

                            if flag_texpaint:
                                # TRELLIS를 GPU에서 잠시 내려 paint(별도 env, ~21GB)용 VRAM 확보
                                meshrecon.pipeline.cpu()
                                clear_gpu_memory()
                                painted_path = os.path.join(capture_dir, "mesh_painted.glb")
                                ref_image = os.path.join(capture_dir, "rgb_masked.png")
                                try:
                                    texpaint_runner.run(output_path, ref_image, painted_path)
                                    output_path = painted_path
                                    print(f"[TexPaint] Saved to {painted_path}")
                                finally:
                                    meshrecon.pipeline.cuda()

                            # HoloLens2에 신호 전송
                            signal = b"1"
                            sock.sendto(signal, (host, 5005))
                            print(f"[UDP] Signal sent to {host}:5005")

                            # local_ip = ock_check.gethostbyname(sock_check.gethostname())
                            # print(f"[Info] GLB available at http://{local_ip}:{HTTP_PORT}/output.glb")

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

                    obj_idx += 1

                    print("[Info] Ready for next reconstruction (Press Space)")
                    # break 제거 - 계속 루프 실행

    except KeyboardInterrupt:
        print("\n[Info] Shutting down...")

    finally:
        if flag_recon_mesh:
            del meshrecon
        if hasattr(hosegmentor, "close"):
            hosegmentor.close()
        del hosegmentor
        clear_gpu_memory()

        sock.close()
        hl2_manager.destroy()
        cv2.destroyAllWindows()
        restore_stdin_cbreak(console_state)


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
