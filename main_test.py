import os
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread
import torch
import gc

## Set HoloLens2 options ##
host = '192.168.50.31'  # HoloLens2 wifi address

# HTTP 서버 설정
HTTP_PORT = 8000

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """GLB 파일을 제공하는 커스텀 HTTP 핸들러"""

    def end_headers(self):
        # CORS 헤더 추가 (필요시)
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        # 로그를 더 간결하게
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
    # GPU 메모리 정리
    clear_gpu_memory()
    print_gpu_memory()

    ###################### HTTP 서버 시작 ######################
    print("\n=== Starting HTTP Server ===")
    print(f"GLB file should be in current directory: {os.getcwd()}")
    print(f"Make sure 'output.glb' exists here!")

    # output.glb 파일 확인
    if os.path.exists('output.glb'):
        file_size = os.path.getsize('output.glb') / (1024 * 1024)  # MB
        print(f"✓ Found output.glb ({file_size:.2f} MB)")
    else:
        print("✗ WARNING: output.glb not found in current directory!")

    # HTTP 서버 시작
    http_thread = Thread(target=start_http_server, daemon=True)
    http_thread.start()

    # 서버가 시작될 때까지 대기
    time.sleep(0.5)

    # 로컬 IP 확인
    import socket as sock_module
    try:
        local_ip = sock_module.gethostbyname(sock_module.gethostname())
        print(f"\n[Info] Server is running!")
        print(f"[Info] GLB URL: http://{local_ip}:{HTTP_PORT}/output.glb")
        print(f"[Info] Or use: http://192.168.50.247:{HTTP_PORT}/output.glb")
    except:
        print(f"\n[Info] Server is running on port {HTTP_PORT}")

    print("\n=== Server Running ===")
    print("Press Ctrl+C to stop\n")

    try:
        # 서버 실행 상태 유지
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Info] Shutting down server...")


if __name__ == '__main__':
    # CUDA 메모리 최적화 설정
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    main()