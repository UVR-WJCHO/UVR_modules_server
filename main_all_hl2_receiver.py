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

사용자 상태 추정(modules/modules_HQS.py):
    python main_all_hl2_receiver.py --hqs                         # 상태·요소를 RGB 창에 표시
    --hqs 는 상태를 HL2_MODE(FOCUS / MOVING / TALKING / NONE 문자열)로 1 초마다 올린다.
    또 WATCH_HR / WATCH_ACTIVITY / WATCH_VAD 를 받아 HQS 패널 값(JSON)을 HL2_UI 로 올린다.
    python main_all_hl2_receiver.py --hqs --start-posture standing
    python main_all_hl2_receiver.py --hqs --general               # 상태 우선순위 없이 점수로 고른다
    python main_all_hl2_receiver.py --hqs --profile output/hqs/profiles/A.json   # A 의 프로파일에 쌓는다
    --hqs 는 패널 값과 모드를 사용자 프로파일(modules_HQS_profile)에 1 초마다 더한다. 파일이 있으면 이어 쌓고,
    없으면 새로 만든다. --profile 을 주지 않으면 output/hqs/profiles/default.json 이다.
    python main_all_hl2_receiver.py --record output/hqs/s1.hl2rec # 원본 패킷 녹화 (+ s1.hl2rec.audio)
    --hqs 는 목소리 판정을 위해 HL2_CONTROL 로 마이크(HL2_AUDIO)를 요청한다. --no-voice 로 끈다.
    HL2_CONTROL 은 켤 스트림 전체를 한 번에 정하므로, 레이어를 요청하는 main_adaptiveMM 과 동시에
    돌리면 서로의 요청을 덮어쓴다.
    python modules/modules_HQS.py --replay output/hqs/s1.hl2rec --gui   # 녹화본 재생
"""

import argparse
import json
import os
import sys
import threading
import time
from queue import Queue

import cv2
import numpy as np
import zmq

# hl2_data_pb2 는 _comm/ 아래에 있음 (main_handtrack.py 의 modules/ 패턴과 동일)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_comm"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
import hl2_data_pb2 as proto

DEFAULT_IDENTITY = b"HL2_DEPTH_RECEIVER"
DEFAULT_KEYWORD = b"HL2DATA"

# depth 컬러맵 고정 범위(가까움=파랑, 멀음=빨강). 프레임 간 비교가 쉽도록 고정.
DEPTH_MIN_MM = 200.0
DEPTH_MAX_MM = 4000.0
SHOW_DEPTH_WINDOWS = False   # hl2_depth / hl2_depth_overlay 창. 요약 문자열은 그대로 낸다
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
    """패킷 intrinsics 를 RGB(w x h) 해상도로 맞춘다. 0 이면 화면크기 기반 대략값.

    intrinsics 가 RGB 가 아니라 depth 격자 해상도로 올 때가 있다(RGB 640x360, depth·K 448x252).
    그대로 쓰면 투영이 주점 쪽으로 쏠린다. 주점이 어느 해상도의 중심에 더 가까운지로 K 가
    정의된 해상도를 고르고, RGB 크기로 비례 조정한다.
    """
    fx, fy, cx, cy = pk.fx, pk.fy, pk.cx, pk.cy
    if fx <= 1 or fy <= 1:
        fx = fy = 0.8 * w
        cx, cy = w / 2.0, h / 2.0
    cands = [(w, h)]
    if pk.depth_width > 0 and pk.depth_height > 0:
        cands.append((pk.depth_width, pk.depth_height))
    nw, nh = min(cands, key=lambda s: abs(cx - s[0] / 2.0) + abs(cy - s[1] / 2.0))
    sx, sy = w / float(nw), h / float(nh)
    return np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float32)


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
    if SHOW_DEPTH_WINDOWS:
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


def consumer_loop(args, hqs=None, rec=None, panel=None, ui=None):
    frame_id = 0
    if hqs is not None:
        from modules_HQS import draw as hqs_draw
    if rec is not None:
        from modules_HQS import REC_HEADER
    while True:
        _, _kw, _src, _fid, data = task_queue.get()
        if rec is not None:
            rec.write(REC_HEADER.pack(time.time(), len(data)) + data)
            rec.flush()
        pkt = proto.HL2SensorPacket()
        pkt.ParseFromString(data)

        rgb = decode_rgb(pkt.image_data)
        depth = decode_depth(pkt.depth_data)
        r = None
        if hqs is not None:
            h, w = rgb.shape[:2] if rgb is not None else (pkt.depth_height or 360, pkt.depth_width or 640)
            r = hqs.update(pkt, rgb, intrinsics(pkt, w, h), depth)
            for e in r.events:
                print(f"[HQS] t={r.t:.2f} {e}", flush=True)
            for ev in panel.on_hqs(r):
                ui.send(ev)
        print(f"[#{frame_id}] " + summarize(pkt, rgb, depth)
              + (f"  hqs={r.state}" if r is not None else ""), flush=True)

        vis = None
        if not args.no_gui and rgb is not None:
            vis = render_overlay(pkt, rgb, depth)
            if r is not None:
                hqs_draw(vis, r)
            cv2.imshow("hl2_rgb", vis)
            cv2.waitKey(1)

        if args.save:
            os.makedirs(args.save, exist_ok=True)
            if rgb is not None:
                cv2.imwrite(f"{args.save}/rgb_{frame_id:05d}.png", vis if vis is not None else rgb)
            if depth is not None:
                cv2.imwrite(f"{args.save}/depth_{frame_id:05d}.png", depth)  # 16bit 원본 보존

        frame_id += 1


AUDIO_CONTROL_PERIOD_S = 5.0   # 기기는 15 s 동안 제어를 못 받으면 마이크를 놓는다(HL2_CONTROL 규약)


def audio_loop(hub, hqs, rec_audio):
    """HL2_AUDIO 를 받아 HQS 의 목소리 판정에 넣고, 녹화 중이면 그대로 남긴다."""
    import hl2_sensors_pb2 as sensproto
    from modules_HQS import REC_HEADER
    n, t_ctrl = 0, 0.0
    while True:
        if time.time() - t_ctrl >= AUDIO_CONTROL_PERIOD_S:   # 끊기지 않게 주기적으로 다시 요청한다
            hub.send(sensproto.HL2Control(send_audio=True).SerializeToString())
            t_ctrl = time.time()
        data = hub.get_latest(timeout=0.5)
        if data is None:
            continue
        if rec_audio is not None:
            rec_audio.write(REC_HEADER.pack(time.time(), len(data)) + data)
            rec_audio.flush()
        a = sensproto.HL2Audio()
        a.ParseFromString(data)
        if a.is_aac or not a.audio:
            continue
        hqs.update_audio(a.audio, a.sample_rate, a.start_timestamp / 1e9, max(1, a.channels))
        n += 1
        if n == 1:
            print(f"[HQS] 오디오 수신 시작 ({a.sample_rate} Hz, {a.channels} ch)", flush=True)


MODE_PERIOD_S = 1.0   # HL2_MODE 를 보내는 주기
MODE_REPEAT_S = 5.0   # 같은 모드가 이어지면 이 간격으로만 다시 보낸다. 그 사이는 NONE


def mode_loop(hub, hqs):
    """HQS 상태를 HL2_MODE 로 1 초마다 보낸다. 상태가 바뀐 순간 그 모드를 보내고, 이어지는 동안은
    MODE_REPEAT_S 마다 다시 보낸다. 나머지 틱과 어느 상태에도 들지 않을 때는 NONE 이다."""
    from modules_HQS import NONE_STATE
    every = round(MODE_REPEAT_S / MODE_PERIOD_S)
    prev, ticks = NONE_STATE, every
    while True:
        cur = hqs.state
        ticks += 1
        if cur != NONE_STATE and (cur != prev or ticks >= every):
            out, ticks = cur, 0
        else:
            out = NONE_STATE
        prev = cur
        hub.send(out.encode())
        if out != NONE_STATE:
            print(f"[HQS] HL2_MODE {out}", flush=True)
        time.sleep(MODE_PERIOD_S)


PROFILE_PERIOD_S = 1.0    # 프로파일에 현재 값을 더하는 주기. 한 번이 1 초 관측이다
PROFILE_SAVE_S = 60.0     # 이 간격과 끝날 때 프로파일 파일에 쓴다


def profile_loop(profile, hqs, panel):
    """HQS 패널 값과 모드를 1 초마다 사용자 프로파일에 더하고 주기적으로 저장한다."""
    t_save = time.time()
    while True:
        time.sleep(PROFILE_PERIOD_S)
        now = time.time()
        profile.add(now, hqs.state, panel.snapshot())
        if now - t_save >= PROFILE_SAVE_S:
            profile.save()
            t_save = now


class UiSender:
    """HQS 패널 이벤트를 HL2_UI 로 올린다. 수신 스레드 여럿이 함께 쓰므로 소켓 하나를 잠가서 쓴다."""

    def __init__(self, hub):
        self.hub = hub
        self.lock = threading.Lock()

    def send(self, ev):
        with self.lock:
            self.hub.send(json.dumps(ev, separators=(",", ":")).encode())
        print(f"[HQS] HL2_UI {ev['type']}={ev['value']} ({ev['source']})"
              + ("  HIGHLIGHT" if ev["highlight"] else ""), flush=True)


def watch_loop(hub, kind, panel, ui):
    """와치 채널 하나(WATCH_HR / WATCH_ACTIVITY / WATCH_VAD)를 받아 패널 이벤트로 바꿔 올린다."""
    while True:
        data = hub.get_latest(timeout=1.0)
        if data is None:
            continue
        try:
            ev = panel.on_watch(kind, json.loads(data))
        except (ValueError, TypeError, AttributeError) as e:      # 깨진 메시지 하나로 채널을 멈추지 않는다
            print(f"[HQS] {kind} 메시지 무시: {e}", flush=True)
            continue
        if ev is not None:
            ui.send(ev)


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

    hqs = rec = panel = ui = profile = None
    hubs = []
    if args.hqs:
        from modules_HQS import HQS, HQSPanel
        from modules_HQS_profile import DEFAULT_PROFILE, HQSProfile, locate
        from hub_client import HubClient, KW_HL2_MODE, KW_HL2_UI, KW_WATCH_HR, KW_WATCH_ACTIVITY, KW_WATCH_VAD
        hqs = HQS(start_posture=args.start_posture, general=args.general)
        panel = HQSPanel()
        profile = HQSProfile(args.profile or DEFAULT_PROFILE, locate())
        print(f"[PROFILE] {profile.path} ({'새로 만듦' if profile.is_new else f'이어 쌓음, {profile.hours():.1f} h'})"
              f"  장소 {profile.where['place']} ({profile.where['tz'] or '서버 시간대'})", flush=True)
        ui_hub = HubClient(args.host, args.port, recv_kw=None, result_kw=KW_HL2_UI, identity=identity + b"_UI")
        ui = UiSender(ui_hub)
        hubs.append(ui_hub)
        print(f"[HQS] ready (start_posture={args.start_posture}, general={args.general})", flush=True)
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        rec = open(args.record, "ab")
        print(f"[REC] {args.record}", flush=True)
    threading.Thread(target=consumer_loop, args=(args, hqs, rec, panel, ui), daemon=True).start()

    audio_hub = None
    if hqs is not None:
        mode_hub = HubClient(args.host, args.port, recv_kw=None, result_kw=KW_HL2_MODE,
                             identity=identity + b"_MODE")
        hubs.append(mode_hub)
        threading.Thread(target=mode_loop, args=(mode_hub, hqs), daemon=True).start()
        threading.Thread(target=profile_loop, args=(profile, hqs, panel), daemon=True).start()
        # 수신 등록은 identity 마다 하나이고 HubClient 는 keyword 를 구분해 주지 않아 채널마다 따로 둔다
        for kw, kind in ((KW_WATCH_HR, "hr"), (KW_WATCH_ACTIVITY, "activity"), (KW_WATCH_VAD, "vad")):
            hub = HubClient(args.host, args.port, recv_kw=kw, identity=identity + b"_" + kw, queue_size=32)
            hubs.append(hub)
            threading.Thread(target=watch_loop, args=(hub, kind, panel, ui), daemon=True).start()
    if hqs is not None and not args.no_voice:
        from hub_client import HubClient, KW_HL2_AUDIO, KW_HL2_CONTROL
        # 오디오는 연속 스트림이라 버리면 안 된다. 수신 등록은 identity 마다 하나라 따로 둔다
        audio_hub = HubClient(args.host, args.port, recv_kw=KW_HL2_AUDIO, result_kw=KW_HL2_CONTROL,
                              identity=identity + b"_AUDIO", queue_size=256)
        rec_audio = open(args.record + ".audio", "ab") if args.record else None
        threading.Thread(target=audio_loop, args=(audio_hub, hqs, rec_audio), daemon=True).start()

    try:
        while True:
            msg = sock.recv_multipart()
            if msg[1] == b"NOTIFY":
                _, _, kw, src, fid = msg
                sock.send_multipart([b"", b"DOWNLOAD", kw, src, fid])
            elif msg[1] == b"DATA_REPLY":
                _, _, kw, src, fid, data = msg
                task_queue.put((None, kw, src, fid, data))
    finally:
        if profile is not None:
            profile.save()
            print(f"[PROFILE] 저장 {profile.path} ({profile.hours():.1f} h)", flush=True)
        if audio_hub is not None:                            # 끝날 때 마이크를 바로 놓게 한다
            import hl2_sensors_pb2 as sensproto
            audio_hub.send(sensproto.HL2Control(send_audio=False).SerializeToString())
            time.sleep(0.2)
            audio_hub.close()
        for hub in hubs:
            hub.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HL2 RGB+Depth receiver with overlay (verify reception)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=37001)
    ap.add_argument("--keyword", default=DEFAULT_KEYWORD.decode())
    ap.add_argument("--identity", default=DEFAULT_IDENTITY.decode())
    ap.add_argument("--no-gui", action="store_true", help="창 없이 콘솔 통계만")
    ap.add_argument("--save", default=None, help="프레임 저장 디렉토리")
    ap.add_argument("--hqs", action="store_true", help="사용자 상태 추정(modules_HQS)을 켠다")
    ap.add_argument("--start-posture", default="seated", choices=("seated", "standing"),
                    help="HQS: 시작할 때 자세. 머리 높이가 상대값이라 필요하다")
    ap.add_argument("--general", action="store_true",
                    help="HQS: 상태 우선순위(STATES 순서) 대신 성립하는 상태 중 점수가 가장 높은 것을 고른다")
    ap.add_argument("--record", default=None,
                    help="원본 HL2DATA 패킷을 이 파일에 이어 쓴다(오디오는 <파일>.audio). modules_HQS.py --replay 로 재생")
    ap.add_argument("--no-voice", action="store_true",
                    help="HQS: 마이크를 요청하지 않는다. voice 요소는 0 이 된다")
    ap.add_argument("--profile", default=None,
                    help="HQS: 사용자 프로파일 JSON. 있으면 이어 쌓고 없으면 만든다(기본 output/hqs/profiles/default.json)")
    args = ap.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nshutting down...")
