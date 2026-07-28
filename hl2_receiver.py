"""
HL2 수신 스크립트 (RGB + Depth + 기타 센서 정보).

comm_hub.py 와 동일한 ZeroMQ 프로토콜(DEALER, RECV_REG -> NOTIFY -> DOWNLOAD ->
DATA_REPLY)을 사용하며, hl2_data_pb2(HL2SensorPacket)로 depth 및 확장 필드까지
파싱/시각화한다. protobuf 정의(hl2_data.proto/_pb2)는 _comm/ 아래에 있다.

실행:
    conda activate wiseui_commu
    cd WiseUIServer
    python hl2_receiver.py                 # GUI 창(RGB/Depth) + 콘솔 통계
    python hl2_receiver.py --no-gui        # 콘솔 통계만 (SSH/headless)
    python hl2_receiver.py --save out_dir  # 프레임을 이미지로 저장하며 확인
    python hl2_receiver.py --host <IP>     # 허브가 다른 PC에 있을 때
"""

import argparse
import os
import sys
import threading
from queue import Queue

import cv2
import numpy as np
import zmq

# hl2_data_pb2 는 _comm/ 아래에 있음 (main_handtrack.py 의 modules/ 패턴과 동일)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_comm"))
import hl2_data_pb2 as proto

DEFAULT_IDENTITY = b"HL2_DEPTH_RECEIVER"
DEFAULT_KEYWORD = b"HL2DATA"

task_queue = Queue(maxsize=10)


def decode_rgb(buf):
    if not buf:
        return None
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decode_depth(buf):
    """16bit depth PNG(mm) -> uint16 HxW. 없으면 None."""
    if not buf:
        return None
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def depth_to_vis(depth):
    """uint16 depth(mm) -> 컬러맵 8bit BGR. 0(hole)은 검정."""
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth == 0] = 0
    return vis


def summarize(pkt, rgb, depth):
    parts = [f"t={pkt.timestamp:.3f}"]
    if rgb is not None:
        parts.append(f"rgb={rgb.shape[1]}x{rgb.shape[0]}")
    else:
        parts.append("rgb=None")
    if pkt.depth_enabled or pkt.depth_data:
        if depth is not None:
            v = depth[depth > 0]
            drange = f"{int(v.min())}-{int(v.max())}mm" if v.size else "empty"
            parts.append(f"depth={depth.shape[1]}x{depth.shape[0]}({depth.dtype},{drange})")
        else:
            parts.append(f"depth=declared {pkt.depth_width}x{pkt.depth_height} but decode failed")
    else:
        parts.append("depth=off")
    parts.append(f"head=({pkt.head_pos_x:.2f},{pkt.head_pos_y:.2f},{pkt.head_pos_z:.2f})")
    parts.append(f"quat=({pkt.head_rot_x:.2f},{pkt.head_rot_y:.2f},{pkt.head_rot_z:.2f},{pkt.head_rot_w:.2f})")
    parts.append(f"gaze_o=({pkt.eye_gaze_origin_x:.2f},{pkt.eye_gaze_origin_y:.2f},{pkt.eye_gaze_origin_z:.2f})")
    parts.append(f"gaze_d=({pkt.eye_gaze_dir_x:.2f},{pkt.eye_gaze_dir_y:.2f},{pkt.eye_gaze_dir_z:.2f})")
    parts.append(f"intr(fx,fy,cx,cy)=({pkt.fx:.1f},{pkt.fy:.1f},{pkt.cx:.1f},{pkt.cy:.1f})")
    parts.append(f"hands R/L joints={len(pkt.right_hand)}/{len(pkt.left_hand)}")
    parts.append(f"pinch L/R={pkt.pinch_left}/{pkt.pinch_right} grab L/R={pkt.grab_left}/{pkt.grab_right}")
    parts.append(f"cam2world={len(pkt.cam_to_world)} pts={len(pkt.depth_points)//3}")
    if pkt.speech:
        parts.append(f"speech='{pkt.speech}'")
    return "  ".join(parts)


def consumer_loop(args):
    frame_id = 0
    while True:
        _, _kw, _src, _fid, data = task_queue.get()
        pkt = proto.HL2SensorPacket()
        pkt.ParseFromString(data)

        rgb = decode_rgb(pkt.image_data)
        depth = decode_depth(pkt.depth_data)
        print(f"[#{frame_id}] " + summarize(pkt, rgb, depth), flush=True)

        if args.save:
            os.makedirs(args.save, exist_ok=True)
            if rgb is not None:
                cv2.imwrite(f"{args.save}/rgb_{frame_id:05d}.png", rgb)
            if depth is not None:
                cv2.imwrite(f"{args.save}/depth_{frame_id:05d}.png", depth)  # 16bit 원본 보존

        if not args.no_gui:
            if rgb is not None:
                cv2.imshow("hl2_rgb", rgb)
            if depth is not None:
                cv2.imshow("hl2_depth", depth_to_vis(depth))
            cv2.waitKey(1)

        frame_id += 1


def run(args):
    identity = args.identity.encode()
    keyword = args.keyword.encode()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, identity)
    sock.connect(f"tcp://{args.host}:{args.port}")

    # RECV_REG: [empty, RECV_REG, KW..., Source, Target]
    sock.send_multipart([b"", b"RECV_REG", keyword, identity, b"ALL"])
    print(f"{identity.decode()} :: connected tcp://{args.host}:{args.port}, kw={keyword.decode()}", flush=True)

    threading.Thread(target=consumer_loop, args=(args,), daemon=True).start()

    while True:
        msg = sock.recv_multipart()
        if msg[1] == b"NOTIFY":
            _, _, kw, src, fid = msg
            sock.send_multipart([b"", b"DOWNLOAD", kw, src, fid])
        elif msg[1] == b"DATA_REPLY":
            _, _, kw, src, fid, data = msg
            task_queue.put((None, kw, src, fid, data))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HL2 RGB+Depth receiver (verify reception)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=37001)
    ap.add_argument("--keyword", default=DEFAULT_KEYWORD.decode())
    ap.add_argument("--identity", default=DEFAULT_IDENTITY.decode())
    ap.add_argument("--no-gui", action="store_true", help="창 없이 콘솔 통계만")
    ap.add_argument("--save", default=None, help="프레임 저장 디렉토리")
    args = ap.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nshutting down...")
