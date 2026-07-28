"""
HL2 수신 스크립트 (RGB + Depth + 기타 센서 정보 + 시각화/overlay).

comm_hub.py 와 동일한 ZeroMQ 프로토콜(DEALER, RECV_REG -> NOTIFY -> DOWNLOAD ->
DATA_REPLY)을 사용하며, hl2_data_pb2(HL2SensorPacket)로 depth 및 확장 필드까지
파싱한다. protobuf 정의(hl2_data.proto/_pb2)는 _comm/ 아래에 있다.

시각화(--no-gui 아닐 때):
  - RGB 이미지에 손 관절(월드->이미지 투영), 시선 crosshair, 정보 텍스트 overlay
  - 정렬 depth 를 고정범위 컬러맵으로 표시 + RGB 위에 blend 한 정합 확인 overlay
  - depth_points(원본 포인트클라우드, sendPointCloud=true 시)를 손과 같은 방식으로 투영
  (투영/overlay 로직은 old/zmq_hl2_receiver.py 를 참고해 반영)

실행:
    conda activate wiseui_commu
    cd WiseUIServer
    python hl2_receiver.py                 # 창(RGB overlay / depth / depth overlay) + 콘솔 통계
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

# depth 컬러맵 고정 범위(가까움=파랑, 멀음=빨강). 프레임 간 비교가 쉽도록 고정.
DEPTH_MIN_MM = 200.0
DEPTH_MAX_MM = 4000.0
# 시선 광선 근/원점(m)
GAZE_NEAR = 0.5
GAZE_FAR = 2.0

task_queue = Queue(maxsize=10)


def decode_rgb(buf):
    if not buf:
        return None
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decode_depth(buf):
    """16bit depth PNG(mm) -> uint16 HxW. 없으면 None. 3채널이면 1채널로."""
    if not buf:
        return None
    d = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d[:, :, 0] if d.ndim == 3 else d


# ---------------------------------------------------------------------------
# 포즈 / 투영 (old/zmq_hl2_receiver.py 참고)
# ---------------------------------------------------------------------------
def pose_from_packet(pk):
    """cam_to_world(4x4, 행우선) 우선. 없으면 head 위치+쿼터니언으로 구성.
    반환 (cam_to_world 4x4 or None, source 문자열)."""
    if len(pk.cam_to_world) == 16:
        return np.array(pk.cam_to_world, np.float32).reshape(4, 4), "cam_to_world"
    q = np.array([pk.head_rot_x, pk.head_rot_y, pk.head_rot_z, pk.head_rot_w], np.float32)
    t = np.array([pk.head_pos_x, pk.head_pos_y, pk.head_pos_z], np.float32)
    if np.allclose(q, 0):
        return None, "none"
    x, y, z, w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], np.float32)
    M = np.eye(4, dtype=np.float32); M[:3, :3] = R; M[:3, 3] = t
    return M, "head_pose"


def intrinsics(pk, w, h):
    """패킷 intrinsics. 0 이면 화면크기 기반 대략값."""
    fx, fy, cx, cy = pk.fx, pk.fy, pk.cx, pk.cy
    if fx <= 1 or fy <= 1:
        fx = fy = 0.8 * w
        cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)


def project_points(K, cam_to_world, P, flip_y=True, forward_neg_z=True):
    """월드 점 P(N,3) -> 이미지 픽셀. HoloLens 카메라(-Z 전방) 규약 기본.
    반환 (uv (N,2), valid (N,))."""
    if P.size == 0 or cam_to_world is None:
        return np.zeros((0, 2)), np.zeros((0,), bool)
    w2c = np.linalg.inv(cam_to_world)
    Ph = np.c_[P, np.ones(len(P))]
    pc = (w2c @ Ph.T).T[:, :3]
    z = -pc[:, 2] if forward_neg_z else pc[:, 2]   # 전방 깊이(양수여야 보임)
    valid = z > 1e-3
    z_safe = np.where(valid, z, 1.0)
    yy = -pc[:, 1] if flip_y else pc[:, 1]          # 이미지 v 는 아래로 증가
    u = K[0, 0] * (pc[:, 0] / z_safe) + K[0, 2]
    v = K[1, 1] * (yy / z_safe) + K[1, 2]
    return np.c_[u, v], valid


def project_one(K, c2w, p):
    uv, ok = project_points(K, c2w, np.asarray(p, np.float32).reshape(1, 3))
    return uv[0], bool(ok[0])


def draw_hand(img, uv, valid, color):
    h, w = img.shape[:2]
    for (u, v), ok in zip(uv, valid):
        if not ok:
            continue
        x, y = int(round(u)), int(round(v))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 4, color, -1)


def draw_gaze(img, K, c2w, pk, color=(0, 255, 255)):   # 노란색(BGR)
    """시선(월드 origin+dir)을 손처럼 이미지에 프로젝션. 바라보는 지점에 크로스헤어."""
    if c2w is None:
        return False
    go = np.array([pk.eye_gaze_origin_x, pk.eye_gaze_origin_y, pk.eye_gaze_origin_z], np.float32)
    gd = np.array([pk.eye_gaze_dir_x, pk.eye_gaze_dir_y, pk.eye_gaze_dir_z], np.float32)
    n = np.linalg.norm(gd)
    if n < 1e-3:
        return False
    gd = gd / n
    nu, nok = project_one(K, c2w, go + GAZE_NEAR * gd)
    fu, fok = project_one(K, c2w, go + GAZE_FAR * gd)
    if not fok:
        return False
    x, y = int(round(fu[0])), int(round(fu[1]))
    if nok:
        cv2.line(img, (int(round(nu[0])), int(round(nu[1]))), (x, y), color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 10, color, 2, cv2.LINE_AA)
    cv2.line(img, (x - 14, y), (x + 14, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y - 14), (x, y + 14), color, 1, cv2.LINE_AA)
    return True


def show_aligned_depth(img, depth):
    """정렬 depth 를 고정범위 컬러맵 + RGB 오버레이로 표시. 요약 문자열 반환."""
    if depth is None:
        return None
    valid = depth > 0
    # 고정 범위 컬러맵: 가까움=파랑, 멀음=빨강 (0.2~4m)
    norm = np.clip((depth.astype(np.float32) - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM), 0, 1)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cm[~valid] = 0
    cm = cv2.dilate(cm, np.ones((3, 3), np.uint8))   # 성긴 점 키우기
    cv2.imshow("hl2_depth", cm)
    # 유효 픽셀만 이미지에 겹쳐서 정렬 확인
    if cm.shape[:2] == img.shape[:2]:
        vm = cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ov = img.copy()
        blend = cv2.addWeighted(img, 0.35, cm, 0.65, 0)
        ov[vm] = blend[vm]
        cv2.imshow("hl2_depth_overlay", ov)
    n = int(valid.sum())
    H, W = depth.shape[:2]
    cov = 100.0 * n / (H * W)
    lo = int(depth[valid].min()) if valid.any() else 0
    hi = int(depth[valid].max()) if valid.any() else 0
    if valid.any():
        ys, xs = np.nonzero(valid)
        bb = f" bbox x {xs.min()}~{xs.max()} / y {ys.min()}~{ys.max()}"
    else:
        bb = " bbox -"
    return f"depth {W}x{H} valid {n} ({cov:.2f}%) ({lo}~{hi}mm){bb}"


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


def render_overlay(pkt, rgb, depth):
    """RGB 위에 손/시선/depth-points/정보 텍스트 overlay. depth 창도 표시. 주석된 이미지 반환."""
    vis = rgb.copy()
    h, w = vis.shape[:2]
    c2w, src = pose_from_packet(pkt)
    K = intrinsics(pkt, w, h)

    rh = np.array(pkt.right_hand, np.float32).reshape(-1, 3) if len(pkt.right_hand) else np.zeros((0, 3))
    lh = np.array(pkt.left_hand, np.float32).reshape(-1, 3) if len(pkt.left_hand) else np.zeros((0, 3))
    ruv, rok = project_points(K, c2w, rh)
    luv, lok = project_points(K, c2w, lh)
    draw_hand(vis, ruv, rok, (0, 255, 0))   # 오른손 녹색
    draw_hand(vis, luv, lok, (0, 0, 255))   # 왼손 빨강
    gaze_ok = draw_gaze(vis, K, c2w, pkt)   # 시선: 노란 크로스헤어

    # 원본 depth 포인트클라우드(sendPointCloud=true 시)를 손과 같은 방식으로 투영
    if len(pkt.depth_points) >= 3 and c2w is not None:
        dp = np.array(pkt.depth_points, np.float32).reshape(-1, 3)
        duv, dok = project_points(K, c2w, dp[::10])   # 10개마다 1개 서브샘플
        for (uu, vv), ok in zip(duv, dok):
            xi, yi = int(round(uu)), int(round(vv))
            if ok and 0 <= xi < w and 0 <= yi < h:
                cv2.circle(vis, (xi, yi), 1, (0, 255, 255), -1)   # 노랑 = 원본 포인트

    dline = show_aligned_depth(rgb, depth)

    info = [
        f"pose: {src}   K fx={K[0,0]:.0f} fy={K[1,1]:.0f}",
        f"head: ({pkt.head_pos_x:.2f},{pkt.head_pos_y:.2f},{pkt.head_pos_z:.2f})",
        f"R hand {len(rh)} (vis {int(rok.sum())})  L hand {len(lh)} (vis {int(lok.sum())})",
        f"gaze: {'vis' if gaze_ok else '-'}  dir=({pkt.eye_gaze_dir_x:.2f},{pkt.eye_gaze_dir_y:.2f},{pkt.eye_gaze_dir_z:.2f})",
        f"pinch R:{pkt.pinch_right} L:{pkt.pinch_left}  grab R:{pkt.grab_right} L:{pkt.grab_left}",
    ]
    if dline:
        info.append(dline)
    for i, line in enumerate(info):
        cv2.putText(vis, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def consumer_loop(args):
    frame_id = 0
    while True:
        _, _kw, _src, _fid, data = task_queue.get()
        pkt = proto.HL2SensorPacket()
        pkt.ParseFromString(data)

        rgb = decode_rgb(pkt.image_data)
        depth = decode_depth(pkt.depth_data)
        print(f"[#{frame_id}] " + summarize(pkt, rgb, depth), flush=True)

        vis = None
        if not args.no_gui and rgb is not None:
            vis = render_overlay(pkt, rgb, depth)
            cv2.imshow("hl2_rgb", vis)
            cv2.waitKey(1)

        if args.save:
            os.makedirs(args.save, exist_ok=True)
            if rgb is not None:
                cv2.imwrite(f"{args.save}/rgb_{frame_id:05d}.png", vis if vis is not None else rgb)
            if depth is not None:
                cv2.imwrite(f"{args.save}/depth_{frame_id:05d}.png", depth)  # 16bit 원본 보존

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
    ap = argparse.ArgumentParser(description="HL2 RGB+Depth receiver with overlay (verify reception)")
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
