"""main_handtrack 의 comm_hub 버전.

기존 main_handtrack.py 는 Hl2Manager(hl2ss 직접 스트리밍)로 RGB/depth 를 받고 UDP 로
결과를 되돌렸다. 이 버전은 통신을 **완전히 comm_hub 형태**로 바꾼다:

  수신: comm_hub(DEALER) 로 HL2DATA(HL2SensorPacket) 구독
        -> RGB(JPEG) + RGB정렬 depth(16bit PNG) + cam_to_world + intrinsics + timestamp
  송신: 결과(ServerResult)를 comm_hub UPLOAD(kw=SERVER_RESULT) 로 되돌림.
        hand pose 가 메인(gesture_idx 는 덤). hand 는 **absolute 3D 관절**(21x3 xyz, meters,
        PV 카메라 프레임)로, aligned depth 에서 얻은 실제 wrist depth + root-relative z 로
        서버에서 lift 한 값이다. 수신 프레임의 PV pose/intrinsics/timestamp 도 echo.

기존 main_handtrack.py 는 그대로 유지되며 이 파일은 별도다.

실행:
    conda activate uvr_integ            # 핸드/제스처 모델용 (torch 등)
    cd WiseUIServer
    python main_handtrack_comm.py --host <브로커IP> --port 37001
"""
import sys
import os
import time
import argparse
from collections import deque
from queue import Queue, Empty
import threading

# WILOR mesh renderer 는 offscreen GL 필요 -> headless(디스플레이 없음)에서도 되게 EGL 기본 사용.
# (모델 import 전에 설정해야 pyrender/OpenGL 이 EGL 로 초기화됨. 이미 지정돼 있으면 존중.)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import zmq

# modules/ 를 top-level 로 import 가능하게 (main_handtrack.py 와 동일)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
# _utils/ (visualize 등) 도 top-level 로
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_utils"))
# protobuf 정의는 _comm/ 아래
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_comm"))
import hl2_data_pb2 as proto


# --- CONFIGURATION ---
GESTURE_CKPT_FILE = "gesture_checkpoint-40.tar"
FLAG_INTERACTION_DETECT = False  # object-aware gesture detection
FLAG_VISUALIZE_DEPTH = True     # True 면 Depth 창을 띄운다 (선택적 시각화)
FLAG_GESTURE = False              # True 면 gesture 인식 수행 (False 면 hand tracking 만)

# comm_hub 브로커 접속
BROKER_HOST = "127.0.0.1"   # comm_hub 브로커 IP
BROKER_PORT = 37001
RECV_KW = b"HL2DATA"
RESULT_KW = b"SERVER_RESULT"
IDENTITY = b"HANDTRACK"

# Front RGB camera parameters (hand tracker 입력 크기)
PV_WIDTH = 640
PV_HEIGHT = 360

DEPTH_MAX_MM = 4000.0  # depth 표시 정규화용
# aligned depth 는 손목 '표면'을 찍지만 실제 wrist 관절은 그보다 안쪽(카메라에서 더 멂).
# 표면 depth 에 이 값을 더해 관절 중심 쪽으로 보정(~손목 두께 절반). 튜닝 가능.
WRIST_DEPTH_OFFSET_MM = 13.0

# Depth image processing frequency
NUM_DEPTH_COUNT = 10  # Process depth once every N RGB frames

# Gesture sequence parameters
SEQ_LEN = 16
THRESHOLD_NUM = 5


# --- comm_hub 클라이언트 (수신 conflate + 결과 UPLOAD) ---
class HubClient:
    """comm_hub DEALER 클라이언트. 수신은 백그라운드 스레드에서 최신 프레임만 유지(conflate),
    결과 UPLOAD 는 별도 소켓으로(스레드 안전) 보낸다."""

    def __init__(self, host, port, recv_kw=RECV_KW, result_kw=RESULT_KW, identity=IDENTITY):
        self.ctx = zmq.Context()
        self.identity = identity
        self.result_kw = result_kw

        # 수신 소켓 (백그라운드 스레드 전용)
        self.rx = self.ctx.socket(zmq.DEALER)
        self.rx.setsockopt(zmq.IDENTITY, identity)
        self.rx.setsockopt(zmq.RCVTIMEO, 500)
        self.rx.connect(f"tcp://{host}:{port}")
        self.rx.send_multipart([b"", b"RECV_REG", recv_kw, identity, b"ALL"])

        # 송신 소켓 (메인 스레드 전용) — 같은 소켓을 두 스레드가 쓰지 않도록 분리
        self.tx = self.ctx.socket(zmq.DEALER)
        self.tx.setsockopt(zmq.IDENTITY, identity + b"_TX")
        self.tx.connect(f"tcp://{host}:{port}")

        self.q = Queue(maxsize=1)   # 최신 1프레임만 (느린 처리 중 오래된 프레임 드롭)
        self._fid = 0
        self._stop = False
        self._thr = threading.Thread(target=self._rx_loop, daemon=True)
        self._thr.start()
        print(f"{identity.decode()} :: connected tcp://{host}:{port}, recv={recv_kw.decode()} result={result_kw.decode()}", flush=True)

    def _rx_loop(self):
        while not self._stop:
            try:
                msg = self.rx.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break   # 소켓/컨텍스트 종료 시
            if msg[1] == b"NOTIFY":
                _, _, kw, src, fid = msg
                self.rx.send_multipart([b"", b"DOWNLOAD", kw, src, fid])
            elif msg[1] == b"DATA_REPLY":
                _, _, kw, src, fid, data = msg
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except Empty:
                        pass
                self.q.put(data)

    def get_latest(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            return None

    def send_result(self, payload):
        self._fid += 1
        self.tx.send_multipart([b"", b"UPLOAD", self.result_kw, self.identity,
                                str(self._fid).encode(), payload])

    def close(self):
        self._stop = True
        self.rx.close(); self.tx.close(); self.ctx.term()


# --- 디코딩 헬퍼 ---
def decode_rgb(buf):
    if not buf:
        return None
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def decode_depth_m(buf, w, h):
    """16bit depth PNG(mm) -> float32 meters, (h,w) 로 resize. 없으면 None."""
    if not buf:
        return None
    d = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[:, :, 0]
    dm = d.astype(np.float32) / 1000.0
    if dm.shape[:2] != (h, w):
        dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_NEAREST)
    return dm


def decode_depth_aligned_mm(buf):
    """RGB-aligned depth PNG -> uint16 (H,W) mm, native PV 해상도(리사이즈 X). 없으면 None."""
    if not buf:
        return None
    d = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d[:, :, 0] if d.ndim == 3 else d


def _sample_wrist_depth_mm(depth_mm, u, v, win=3):
    """(u,v) 주변 window 의 유효(>0) depth 중 최근접(frontmost=손 표면) mm. 없으면 None."""
    H, W = depth_mm.shape[:2]
    u = int(round(u)); v = int(round(v))
    best = 0
    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            x, y = u + dx, v + dy
            if 0 <= x < W and 0 <= y < H:
                m = int(depth_mm[y, x])
                if m > 0 and (best == 0 or m < best):
                    best = m
    return best if best > 0 else None


def lift_pose_cam3d(outs_uvd, depth_mm, fx, fy, cx, cy, wrist_mm_fallback=None):
    """RGB uvd(21x3: u,v @640x360, z=root-relative mm) + aligned depth(mm, PV 해상도)
    -> absolute 21x3 xyz (meters, PV 카메라 프레임 = OpenCV: +X右 +Y下 +Z앞).
    wrist(관절0) 픽셀의 aligned depth 로 실제 wrist depth 를 얻고, root-relative z 로 나머지를 lifting.
    wrist depth 샘플 실패 시 wrist_mm_fallback(직전 취득값)으로 대체.
    반환 (xyz 21x3 or None, fresh_wrist_mm or None). 둘 다 없으면 (None, None)."""
    H, W = depth_mm.shape[:2]
    sx = W / float(PV_WIDTH); sy = H / float(PV_HEIGHT)   # 640x360 -> PV 해상도로 환산
    zw_fresh = _sample_wrist_depth_mm(depth_mm, outs_uvd[0, 0] * sx, outs_uvd[0, 1] * sy)
    zw = zw_fresh if zw_fresh is not None else wrist_mm_fallback
    if zw is None:
        return None, None
    zc = zw + WRIST_DEPTH_OFFSET_MM   # 손목 표면 -> 관절 중심 보정
    out = np.zeros((21, 3), np.float32)
    for j in range(21):
        u = outs_uvd[j, 0] * sx
        v = outs_uvd[j, 1] * sy
        Z = (zc + (outs_uvd[j, 2] - outs_uvd[0, 2])) / 1000.0   # mm -> m
        out[j, 0] = (u - cx) / fx * Z
        out[j, 1] = (v - cy) / fy * Z
        out[j, 2] = Z
    return out, zw_fresh


def build_result(pkt, hand_flat, gesture_idx):
    """수신 프레임의 pose/intrinsics/timestamp 를 echo 해서 ServerResult 로 직렬화.
    hand pose 가 메인, gesture_idx 는 덤(-1=없음)."""
    r = proto.ServerResult()
    r.timestamp = pkt.timestamp
    r.cam_to_world.extend(pkt.cam_to_world)      # PV pose (16) echo
    r.fx = pkt.fx; r.fy = pkt.fy; r.cx = pkt.cx; r.cy = pkt.cy
    r.hand.extend([float(x) for x in hand_flat])
    r.gesture_idx = int(gesture_idx)
    return r.SerializeToString()


# --- MAIN ---
def main():
    ap = argparse.ArgumentParser(description="HL2 handtrack over comm_hub")
    ap.add_argument("--host", default=BROKER_HOST, help="comm_hub 브로커 IP")
    ap.add_argument("--port", type=int, default=BROKER_PORT)
    args = ap.parse_args()

    # 무거운 모델 import 는 여기서 (모듈 import 를 가볍게 유지 -> comm 계층 테스트 가능)
    from modules_hand import HandTracker_onnx
    from modules_gesture import GestureClassfier
    from modules_obj import ObjTracker
    from visualize import draw_2d_skeleton   # _utils/visualize.py

    # 1. Initialize Models
    try:
        hand_tracker = HandTracker_onnx()   # WILOR-ONNX
        hand_tracker.warmup(np.zeros((PV_HEIGHT, PV_WIDTH, 3), dtype=np.uint8))

        track_gesture = GestureClassfier(
            ckpt=f"./pretrained/{GESTURE_CKPT_FILE}",
            seq_len=SEQ_LEN, model_opt=1) if FLAG_GESTURE else None
        track_obj = ObjTracker() if FLAG_INTERACTION_DETECT else None
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    # 2. comm_hub 연결
    hub = HubClient(args.host, args.port)

    # 3. Visualization
    cv2.namedWindow('Prompt')
    cv2.resizeWindow(winname='Prompt', width=500, height=500)

    # 4. State
    idx_depth = 0
    flag_cooldown = False
    t_cooldown = 0.0
    queue_righthand = deque([], maxlen=SEQ_LEN)
    prev_gesture = None
    gesture_cnt = 0
    valid_gesture = None
    valid_gesture_idx = -1
    debug_pose = np.ones((21, 3))
    last_wrist_mm = None   # 직전 성공한 wrist depth(mm) — 샘플 실패 시 폴백

    try:
        while True:
            idx_depth = (idx_depth + 1) % (NUM_DEPTH_COUNT + 1)
            flag_depth = (NUM_DEPTH_COUNT > 0 and idx_depth == 0)

            # 5. 수신 (comm_hub, 최신 프레임)
            data = hub.get_latest(timeout=1.0)
            if data is None:
                continue
            pkt = proto.HL2SensorPacket()
            pkt.ParseFromString(data)

            color = decode_rgb(pkt.image_data)
            if color is None:
                continue
            depth = decode_depth_m(pkt.depth_data, PV_WIDTH, PV_HEIGHT) if flag_depth else None
            depth_mm = decode_depth_aligned_mm(pkt.depth_data)   # lift 용 (native PV 해상도, mm)
         

            cv2.imshow('RGB', color)
            if FLAG_VISUALIZE_DEPTH and depth is not None:
                depth_norm = np.clip(depth / (DEPTH_MAX_MM / 1000.0), 0, 1)
                depth_vis = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                depth_vis[depth <= 0] = 0   # 무효 depth 는 검정
                cv2.imshow('Depth', depth_vis)
            cv2.waitKey(1)

            color_resized = cv2.resize(color, dsize=(PV_WIDTH, PV_HEIGHT), interpolation=cv2.INTER_AREA)


            # 6. Hand
            outs = hand_tracker.run(np.copy(color_resized))  # uvd_right
            if not isinstance(outs, np.ndarray):
                valid_gesture = None
                valid_gesture_idx = -1
                gesture_cnt = 0
                # 손 미검출: 더미 pose + -1 (pose/ts/intrinsics 는 echo)
                hub.send_result(build_result(pkt, debug_pose.flatten(), -1))
                continue

            # 7-9. Gesture recognition (optional, FLAG_GESTURE)
            if FLAG_GESTURE:
                # 7. Gesture sequence
                angle_label = track_gesture._compute_ang_from_joint(outs)
                data_seq = np.concatenate([outs.flatten(), angle_label])
                queue_righthand.append(data_seq)

                # 8. Interaction (object-aware)
                flag_gesture = False
                if FLAG_INTERACTION_DETECT and flag_depth and depth is not None and track_obj is not None:
                    uv_wrist = outs[0, :2].astype(int)
                    if 0 <= uv_wrist[1] < PV_HEIGHT and 0 <= uv_wrist[0] < PV_WIDTH:
                        d_wrist = depth[uv_wrist[1], uv_wrist[0]]
                        obj_nearby_list = track_obj.detect_objs_no_cnt(color_resized, depth, d_wrist)
                        if len(obj_nearby_list) > 0:
                            palm_points_2d = outs[[0, 4, 8, 12, 16, 20], :2]
                            palm_uv = np.mean(palm_points_2d, axis=0)
                            for obj_nearby in obj_nearby_list:
                                cx, cy = (obj_nearby[0] + obj_nearby[2]) // 2, (obj_nearby[1] + obj_nearby[3]) // 2
                                distance = np.sqrt((cx - palm_uv[0]) ** 2 + (cy - palm_uv[1]) ** 2)
                                if distance < 70.0:
                                    flag_gesture = True
                                    break
                else:
                    flag_gesture = True

                # 9. Gesture classifier
                gesture = None
                if flag_gesture and len(queue_righthand) == SEQ_LEN:
                    gesture_idx, gesture = track_gesture.run(queue_righthand)
                else:
                    gesture_idx = -1

                if prev_gesture == gesture and gesture != 'Natural' and gesture is not None:
                    gesture_cnt += 1
                else:
                    gesture_cnt = 0
                prev_gesture = gesture

                if gesture_cnt > THRESHOLD_NUM:
                    valid_gesture = gesture
                    valid_gesture_idx = gesture_idx
                    gesture_cnt = 0
                elif gesture_cnt == 0:
                    valid_gesture = None
                    valid_gesture_idx = -1
            else:
                # gesture 인식 off: hand tracking 만, gesture 는 없음
                valid_gesture = None
                valid_gesture_idx = -1

            # 10. Visualization
            vis_color = draw_2d_skeleton(color_resized.copy(), outs[:, :2])
            if valid_gesture is not None and valid_gesture != "Natural":
                cv2.putText(vis_color, f'{valid_gesture.upper()}', org=(20, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
            cv2.imshow("Prompt", vis_color)
            cv2.waitKey(1)

            # 11. 결과 return — aligned depth 로 lift 한 absolute 3D hand pose(메인) + gesture_idx(덤)
            hand3d = None
            if depth_mm is not None:
                # wrist depth 샘플 실패 시 직전 취득값(last_wrist_mm)으로 폴백
                hand3d, zw_fresh = lift_pose_cam3d(outs, depth_mm, pkt.fx, pkt.fy, pkt.cx, pkt.cy,
                                                   last_wrist_mm)
                if zw_fresh is not None:
                    last_wrist_mm = zw_fresh   # 성공 시에만 갱신(스케일 drift 방지)

            if time.time() - t_cooldown > 0.5:
                flag_cooldown = False

            if not flag_cooldown and valid_gesture is not None and valid_gesture != "Natural":
                out_idx = valid_gesture_idx
                flag_cooldown = True
                t_cooldown = time.time()
                print(f"Sending gesture: {valid_gesture}")
            else:
                out_idx = -1

            # lift 실패(손목 depth 없음) 시 더미(전부 1.0) = "유효하지 않음" 표시
            hand_out = hand3d if hand3d is not None else debug_pose
            hub.send_result(build_result(pkt, hand_out.flatten(), out_idx))

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    finally:
        print("Cleaning up resources...")
        hub.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
