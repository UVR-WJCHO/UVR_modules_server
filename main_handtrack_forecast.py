"""main_handtrack 의 forecast 버전.

기존 `main_handtrack.py` 는 프레임마다 WiLoR 를 돌려 **그 프레임의** 자세를 돌려준다.
그 자세는 HL2 에 도착할 때 이미 낡아 있다. 이 파일은 같은 입력에서 **여러 horizon 의
forecast grid** 를 만들어 보낸다. HL2 는 렌더 시점의 pose age 를 재서 인접한 두 forecast
를 보간해 그린다 (계획서 §0.1).

    수신: comm_hub(DEALER) 로 HL2DATA(HL2SensorPacket) 구독
    송신: HandForecast 를 kw=HAND_FORECAST 로 UPLOAD (기존 SERVER_RESULT 와 별도 채널)

**기존 파일을 하나도 고치지 않는다.** `main_handtrack.py`, `modules/`,
`_comm/hl2_data.proto` 는 읽기만 한다. forecast 메시지는 `_comm/hl2_forecast.proto` 로
따로 두었고, 양손 추론은 tracker 의 공개/내부 메서드를 밖에서 호출해 처리한다.

실행 (인자 없이 기본값으로 동작한다):
    conda activate uvr_integ
    python main_handtrack_forecast.py

기본값은 브로커 127.0.0.1:37001 과 mixed_big_wilor/seed0 의 forecast 모델이다.
다른 브로커나 checkpoint 를 쓸 때만 --host / --port / --onnx 를 넘긴다.
"""
import argparse
import contextlib
import os
import sys
import time
from queue import Empty, Queue
import threading

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import zmq

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "modules"))
sys.path.insert(0, os.path.join(_ROOT, "_comm"))
import hl2_data_pb2 as proto                     # noqa: E402  (읽기 전용)
from hub_client import (                       # noqa: E402
    HubClient, add_broker_args, KW_HAND_FORECAST)

from modules.delay_nowcasting.data.canonical_schema import (                # noqa: E402
    BONES, NUM_JOINTS, WRIST)
from modules.delay_nowcasting.deployment import forecast_packet as fp         # noqa: E402
from modules.delay_nowcasting.deployment.onnx_runner import (                 # noqa: E402
    DEFAULT_GRID_MS, ForecastRunner)
from modules.delay_nowcasting.deployment.server_history import HistoryStore   # noqa: E402

# comm_hub 접속. 주소와 keyword 는 _comm/hub_client.py 가 갖는다
IDENTITY = b"HANDFORECAST"

PV_WIDTH, PV_HEIGHT = 640, 360
# 배포 모델 (mixed_25d). 입력·출력이 WiLoR 원본과 같은 2.5D 다 — 관절마다
#   (u_n, v_n, rel_z) = (x/z, y/z, 손목 기준 상대 depth).
# 손목 절대 depth 를 쓰지 않으므로 그 채널의 튐이 들어오지 않는다. HL2 기록에서
# 프레임간 손목 depth 가 최대 458 mm 튀었고 그 튐이 3D 튐과 상관 +0.93 이었다.
# 공용 데이터셋 199 ms 기준 hold 48.0 -> 38.4 mm, 전체로는 hold 대비 31.6% 개선.
# bone_2d loss 포함. ablation 의 평균 MPJPE 로는 전 구간 +-0.5% 로 차이가 없었으나,
# 실사용 체감은 따로 확인 중이다. 뺀 모델은 mixed_25d_nobone 에 있다.
# 관절별 base 이득(base_gain)은 유지한다. 정량 이득은 199 ms 에서 1.5~3.4% 로
# 작지만, 손끝이 크게 튀는 프레임이 소수라 평균 MPJPE 가 그 거동을 못 잡는다.
# 이름으로 고를 수 있는 forecast 모델. --model 로 고르고, 목록에 없는 것은 --onnx 로 준다.
#   mixed3 : DexYCB + HOT3D + HOI4D 세 도메인 + 기하 증강. 배포 기본값이다. 배포 기기는
#            어느 촬영 조건과도 일치하지 않으므로 단일 도메인 모델은 쓰지 않는다 — 다른
#            도메인으로 옮겨가지 않는다는 것을 교차 평가에서 확인했다.
#   mixed2 : DexYCB + HOT3D 두 도메인, 증강 이전. 2026-08-30 까지 배포하던 모델이며
#            비교용으로 남긴다.
# 주의: 둘 다 아직 1 seed 다.
FORECAST_MODELS = {
    "mixed3": "pretrained/forecast/mixed3.onnx",
    "mixed2": "pretrained/forecast/mixed2.onnx",
}
DEFAULT_FORECAST_MODEL = "mixed3"
DEFAULT_FORECAST_ONNX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    FORECAST_MODELS[DEFAULT_FORECAST_MODEL])


# 시각화. 손목 + 손가락당 4 관절 순서(canonical 21-joint).
BONES = tuple((0, 1 + 4 * f) for f in range(5)) + tuple(
    (1 + 4 * f + j, 2 + 4 * f + j) for f in range(5) for j in range(3))
ANCHOR_COLOR = (60, 220, 60)                       # BGR, 추적된 현재 자세
HORIZON_COLORS = ((0, 200, 255), (0, 120, 255), (0, 40, 230))   # 가까운 -> 먼 horizon


def project_to_image(pose_25d, intrinsics, scale_x, scale_y):
    """2.5D -> PV 이미지 좌표. u_n, v_n 이 곧 광선이라 depth 도 카메라 pose 도 필요 없다."""
    fx, fy, cx, cy = intrinsics
    p = np.asarray(pose_25d, np.float64)
    return np.stack([(fx * p[:, 0] + cx) / scale_x,
                     (fy * p[:, 1] + cy) / scale_y], axis=-1)


def draw_skeleton(image, uv, color, thickness=2, radius=3):
    """color 는 단일 BGR 이거나 관절별 (21, 3) 배열이다. 뼈는 양 끝 색의 평균으로 긋는다."""
    colors = np.broadcast_to(np.asarray(color, np.float64), (len(uv), 3))
    finite = np.isfinite(uv).all(axis=1)
    for a, b in BONES:
        if finite[a] and finite[b]:
            cv2.line(image, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)),
                     tuple(((colors[a] + colors[b]) / 2).astype(int).tolist()),
                     thickness, cv2.LINE_AA)
    for i in np.flatnonzero(finite):
        cv2.circle(image, tuple(uv[i].astype(int)), radius,
                   tuple(colors[i].astype(int).tolist()), -1, cv2.LINE_AA)


def save_poses(path, poses) -> None:
    """프레임별 자세 기록을 npz 하나로 낸다. 오프라인 재현용이다.

    없는 값(검출 실패, history 미달)은 NaN 으로 채워 모양을 고정한다. 그래야 어떤
    프레임이 왜 빠졌는지도 같이 남는다.
    """
    if not poses:
        return
    keys = [k for k in poses[0] if k != "reason"]
    np.savez_compressed(path, reason=np.array([p["reason"] for p in poses]),
                        **{k: np.stack([p[k] for p in poses]) for k in keys})


def render_forecast(canvas, anchor, grid, horizons_ms, vis_horizons,
                    intrinsics, scale_x, scale_y, weight=None):
    """추적 자세 위에 선택한 horizon 의 예측 자세를 겹쳐 그린다. 좌표는 2.5D 다.

    `weight` 를 주면 관절 색이 혼합 비중을 나타낸다 — WiLoR 를 그대로 쓰는 관절은
    anchor 와 같은 초록, 예측을 그대로 쓰는 관절은 horizon 색, 사이는 그 사이 색이다.
    """
    if anchor is not None:
        draw_skeleton(canvas, project_to_image(anchor, intrinsics, scale_x, scale_y),
                      ANCHOR_COLOR, 2, 4)
    if grid is None:
        return
    base = np.asarray(ANCHOR_COLOR, np.float64)
    for n, h in enumerate(vis_horizons):
        i = int(np.argmin(np.abs(np.asarray(horizons_ms) - h)))
        far = np.asarray(HORIZON_COLORS[n % len(HORIZON_COLORS)], np.float64)
        color = far if weight is None else base + (far - base) * np.asarray(weight)[:, None]
        draw_skeleton(canvas, project_to_image(grid[i], intrinsics, scale_x, scale_y),
                      color, 1, 2)


# ==========================================================================
# 웹캠 입력 (임시). HL2 없이 파이프라인을 돌려보기 위한 것이다.
# 제거할 때는 이 블록과 --source 인자, main() 의 source 분기만 지우면 된다.
# 메인 루프는 손대지 않는다 — 웹캠 프레임을 같은 HL2SensorPacket 으로 만들어 넣는다.
# ==========================================================================
@contextlib.contextmanager
def _muted_stderr():
    """C 레벨 stderr 를 잠시 /dev/null 로 돌린다.

    이 웹캠의 MJPG 스트림은 restart marker 앞에 여분 바이트가 붙어 있어 libjpeg 가
    "Corrupt JPEG data: N extraneous bytes" 를 매 프레임 찍는다. 디코드 자체는
    정상이다(read 가 True 를 돌려주고 프레임도 멀쩡하다). 파이썬 warnings 로는
    안 잡히므로 fd 를 직접 막는다. cap.read() 호출 구간에만 건다.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


class WebcamSource:
    """웹캠 프레임을 HL2SensorPacket 으로 포장해 HubClient 와 같은 인터페이스로 낸다.

    forecast 가 2.5D 라 depth 도 카메라 pose 도 안 쓴다. 그래서 웹캠으로도 실제와
    같은 경로가 그대로 돈다 — 예전처럼 가짜 depth 를 지어내지 않는다.

    다만 intrinsics 는 수평 화각 WEBCAM_FOV_DEG 로 역산한 값이라 실제와 다르다.
    u_n = (u - cx) / fx 이므로 fx 가 틀리면 입력 배율이 그만큼 어긋난다. 떨림이나
    발산 여부를 보는 데는 지장이 없지만 절대 정확도는 이 값에 좌우된다.
    """

    WEBCAM_FOV_DEG = 60.0

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(index)
        # MJPG 를 지정하지 않으면 OpenCV 가 무압축 YUYV 를 고른다. 1280x720 YUYV 는
        # 프레임당 1.8 MB 라 USB 대역폭에서 8 fps 에 막힌다 — 실측 135.9 ms.
        # MJPG 로 바꾸면 같은 해상도에서 32.2 ms (31 fps) 다.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise SystemExit(f"웹캠 {index} 를 열 수 없다")
        with _muted_stderr():
            ok, frame = self.cap.read()
        if not ok:
            raise SystemExit(f"웹캠 {index} 에서 프레임을 읽을 수 없다")
        h, w = frame.shape[:2]
        focal = (w / 2.0) / np.tan(np.radians(self.WEBCAM_FOV_DEG / 2.0))
        self.intrinsics = (focal, focal, w / 2.0, h / 2.0)
        self._start = time.perf_counter()
        self.n_arrived = 0
        self.n_dropped = 0
        print(f"webcam {index}: {w}x{h}, focal {focal:.1f} px (화각 "
              f"{self.WEBCAM_FOV_DEG:.0f}도 가정)", flush=True)

    def get_latest(self, timeout: float = 1.0):
        with _muted_stderr():
            ok, frame = self.cap.read()
        if not ok:
            return None
        self.n_arrived += 1
        packet = proto.HL2SensorPacket()
        packet.image_data = cv2.imencode(".jpg", frame)[1].tobytes()
        # intrinsics 가 어느 해상도 기준인지 알려 준다. 웹캠은 프레임 해상도 그대로다.
        packet.depth_width, packet.depth_height = frame.shape[1], frame.shape[0]
        packet.fx, packet.fy, packet.cx, packet.cy = self.intrinsics
        packet.timestamp = time.perf_counter() - self._start
        return packet.SerializeToString()

    def send(self, payload):
        """웹캠 모드에서는 보낼 곳이 없다. 시각화로만 확인한다."""

    def close(self):
        self.cap.release()


# 관절별 anchor/예측 혼합. 정지한 관절은 WiLoR 자세가 이미 잘 맞으므로 예측이
# 얹어 주는 것이 오차뿐이다. 관절마다 최근 속도를 재서 느리면 anchor 쪽으로,
# 빠르면 예측 쪽으로 간다.
#
# 속도는 **화면 픽셀 기준**이다 (PV 640x360 좌표계). depth 도 3D 도 개입하지 않는다.
# 문턱은 WiLoR history 실측 분포의 사분위수를 픽셀로 옮긴 값이다 — 관절별 프레임간
# 속도 147,808 쌍에서 p25 0.18 /s, p75 0.72 /s 였고 초점거리를 곱하면 아래 값이 된다.
# p25 아래는 사실상 정지로 보고 anchor 를 그대로 쓰고, p75 위는 예측을 그대로 쓴다.
BLEND_SPEED_PX = (100.0, 500.0)      # PV px/s. 이 아래는 anchor, 이 위는 예측


def blend_with_anchor(grid, history, times, intrinsics, scale_x, scale_y):
    """관절별 화면 속도로 anchor 와 예측을 섞는다. ((H,21,3), (21,)) 를 돌려준다.

    속도는 history 전체의 프레임간 픽셀 속도 중앙값으로 잰다. 2 프레임 차분은
    잡음에 끌려가 정지한 관절을 움직이는 것으로 보이게 한다.
    """
    dt = np.diff(times) / 1000.0
    ok = dt > 1e-6
    if not ok.any():
        return np.asarray(grid, np.float64), np.ones(grid.shape[1])
    uv = np.stack([project_to_image(h, intrinsics, scale_x, scale_y) for h in history])
    speed = np.linalg.norm(np.diff(uv, axis=0)[ok], axis=-1) / dt[ok, None]   # (N-1, 21)
    low, high = BLEND_SPEED_PX
    weight = np.clip((np.median(speed, axis=0) - low) / (high - low), 0.0, 1.0)
    anchor = history[-1]
    return anchor + (np.asarray(grid, np.float64) - anchor) * weight[:, None], weight


# 전송 시 축 규약. 서버 내부는 OpenCV 규약(+X 우, +Y 하, +Z 앞)이라 u_n = x/z 이지만,
# 기기는 +X 가 왼쪽인 좌표계로 받는다. 실기기에서 이 부호를 뒤집어야 손이 제대로
# 놓이는 것을 확인했다 (뒤집기 전에는 좌우가 반전됐다).
#
# 기기 쪽 변환 어디에서 반사가 들어가는지는 확정하지 못했다. Unity 의 cam_to_world 는
# 회전부 det 가 -1 이라 반사를 품고 있고(예전 진단에서 확인), 3D 를 보내던 시절에는
# 서버가 그 변환을 직접 해서 상쇄됐다. 지금은 기기가 하므로 규약을 맞춰 내보낸다.
# 이 값은 서버 내부(history, 시각화)에는 영향을 주지 않는다 — 전송 직전에만 곱한다.
SEND_AXES = np.array([1.0, 1.0, 1.0])      # 반전 끔. 되돌리려면 첫 값을 -1.0 로.


# forecast 출력 EMA. 1.0 이면 평활 없음.
#
# 끈 이유: EMA 는 프레임 단위로 섞으므로 지연이 프레임률에 그대로 비례한다.
# alpha 0.6 이면 지연이 대략 (1-alpha)/alpha * dt 이고, dt 가 135 ms 면 90 ms 다.
# 예측이 앞서 나가라고 만든 것을 뒤로 끌어당기는 셈이 된다. 0.6 은 3D 표현에서
# 15~30 fps 를 전제로 고른 값이었고, 2.5D 로 바꾼 지금은 다시 정해야 한다.
FORECAST_EMA_ALPHA = 1.0


class ForecastSmoother:
    """track 별·horizon 별로 직전 출력을 들고 지수평균을 낸다.

    history 가 reset 되면(손을 놓쳤다가 다시 잡으면) 이전 값을 버린다. 그러지 않으면
    끊긴 구간을 가로질러 섞여 자세가 튄다.
    """

    def __init__(self, alpha: float = FORECAST_EMA_ALPHA):
        self.alpha = alpha
        self._last: dict[str, np.ndarray] = {}

    def reset(self, key: str) -> None:
        self._last.pop(key, None)

    def __call__(self, key: str, grid: np.ndarray) -> np.ndarray:
        previous = self._last.get(key)
        smoothed = (grid if previous is None or previous.shape != grid.shape
                    else self.alpha * grid + (1.0 - self.alpha) * previous)
        self._last[key] = smoothed
        return smoothed


def to_25d(detection, intrinsics, scale_x, scale_y):
    """WiLoR 검출 -> (21, 3) = (u_n, v_n, rel_z). 학습 표현(build_25d.py)과 같다.

    depth 를 쓰지 않는다. u_n, v_n 은 2D 키포인트를 초점거리로 정규화한 것이고
    rel_z 는 WiLoR 의 root-relative z 다. 손목의 rel_z 는 0 이 된다.
    """
    fx, fy, cx, cy = intrinsics
    uv = np.asarray(detection["joints_2d"], np.float64)
    root = np.asarray(detection["root_relative"], np.float64)
    return np.stack([(uv[:, 0] * scale_x - cx) / fx,
                     (uv[:, 1] * scale_y - cy) / fy,
                     root[:, 2] - root[WRIST, 2]], axis=-1)


def detect_hands(tracker, frame):
    """프레임에서 검출된 손을 **모두** 돌려준다.

    `tracker.process()` 는 손 하나만 돌려주는데(오른손 우선), forecast 는 손별 history 가
    필요하다. 공유 코드를 고치지 않기 위해 tracker 내부 메서드를 밖에서 호출한다 —
    연구 코드의 `wilor_infer.infer_frame` 과 같은 방식이다.
    """
    import torch

    from handtracker_onnx.wilor_onnx_utils import cam_crop_to_full

    boxes, classes = tracker.detect(frame)
    if boxes is None:
        return []

    # 손별 패치를 모아 **한 번의 batch 로** pose head 를 돌린다. batch 1 을 두 번 부르면
    # 커널 런치와 메모리 전송이 두 벌 나간다 (실측 15.30 ms vs batch 2 한 번 11.66 ms).
    patches, meta, seen = [], [], set()
    for box, cls in zip(boxes, classes):
        is_right = int(cls)
        handedness = "RIGHT" if is_right else "LEFT"
        if handedness in seen:
            continue
        seen.add(handedness)
        patch, center, bbox_size = tracker._preprocess_hand_patch(
            frame, box, do_flip=(is_right == 0))
        patches.append(patch)
        meta.append((handedness, is_right, center, bbox_size))
    if not patches:
        return []

    outputs = tracker.session.run(
        None, {tracker.input_name: np.concatenate(patches, axis=0)})
    pred_cam_all = outputs[0].astype(np.float32)
    joints_all = outputs[1].astype(np.float32)

    focal = float(tracker.cfg.EXTRA.FOCAL_LENGTH / tracker.cfg.MODEL.IMAGE_SIZE
                  * max(frame.shape[1], frame.shape[0]))
    image_size = np.array([[frame.shape[1], frame.shape[0]]], np.float32)

    results = []
    for n, (handedness, is_right, center, bbox_size) in enumerate(meta):
        # 좌우 되돌리기는 손마다 다르므로 batch 출력에서 개별로 처리한다.
        pred_cam = pred_cam_all[n:n + 1].copy()
        joints = joints_all[n].copy()
        pred_cam[0, 1] *= float(2 * is_right - 1)
        if is_right == 0:
            joints[:, 0] = -joints[:, 0]

        cam_t = cam_crop_to_full(
            torch.from_numpy(pred_cam),
            torch.from_numpy(np.array([center], np.float32)),
            torch.from_numpy(np.array([bbox_size], np.float32)),
            torch.from_numpy(image_size),
            torch.tensor(focal, dtype=torch.float32)).cpu().numpy()[0]
        joints_2d = tracker._project_full_img(
            joints, cam_t, focal, image_size[0])
        results.append({"handedness": handedness, "joints_2d": joints_2d.astype(np.float64),
                        "root_relative": joints.astype(np.float64)})
    return results


def main():
    ap = argparse.ArgumentParser(description="HL2 handtrack forecast over comm_hub")
    add_broker_args(ap)
    ap.add_argument("--model", choices=sorted(FORECAST_MODELS), default=DEFAULT_FORECAST_MODEL,
                    help="forecast 모델 (기본: %(default)s). --onnx 를 주면 무시된다")
    ap.add_argument("--onnx", default=None,
                    help="export_onnx 로 만든 모델 경로. --model 대신 직접 지정할 때 쓴다")
    ap.add_argument("--device-id", default="hl2-0")
    ap.add_argument("--history-length", type=int, default=8)
    ap.add_argument("--max-gap-ms", type=float, default=250.0)
    ap.add_argument("--horizons-ms", nargs="*", type=float, default=list(DEFAULT_GRID_MS))
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--vis", action="store_true", help="추적 자세와 예측 자세를 창에 띄운다")
    ap.add_argument("--save-vis", default=None, metavar="DIR",
                    help="시각화 프레임을 이 디렉터리에 저장한다 (--vis 없이도 동작)")
    ap.add_argument("--save-pose", default=None, metavar="FILE.npz",
                    help="프레임별 자세(history/anchor/예측/depth)를 npz 로 저장한다")
    ap.add_argument("--save-every", type=int, default=1,
                    help="몇 프레임마다 저장할지 (기본: %(default)s)")
    ap.add_argument("--save-max", type=int, default=200,
                    help="최대 저장 장수. 넘으면 저장을 멈춘다 (기본: %(default)s)")
    # 웹캠 입력 (임시). 제거할 때 이 두 인자와 main() 의 분기, WebcamSource 를 지운다.
    ap.add_argument("--source", default="hub", choices=["hub", "webcam"],
                    help="hub = comm_hub(HL2), webcam = 로컬 카메라로 떨림만 확인")
    ap.add_argument("--webcam-index", type=int, default=0)
    ap.add_argument("--ema-alpha", type=float, default=FORECAST_EMA_ALPHA,
                    help="forecast 출력 EMA 계수. 1.0 이면 평균하지 않는다 "
                         "(기본: %(default)s)")
    ap.add_argument("--diag", action="store_true",
                    help="anchor 안정성과 예측 증폭을 진단 출력한다")
    ap.add_argument("--vis-horizons-ms", nargs="*", type=float, default=[66, 133, 199],
                    help="시각화할 horizon. 전부 그리면 겹쳐서 안 보인다")
    args = ap.parse_args()

    if args.onnx is None:
        args.onnx = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 FORECAST_MODELS[args.model])
    if not os.path.exists(args.onnx):
        raise SystemExit(
            f"forecast 모델이 없다: {args.onnx}\n"
            "  export_onnx 로 먼저 만들거나 --onnx 로 경로를 지정하라:\n"
            "  python -m modules.delay_nowcasting.deployment.export_onnx "
            "--checkpoint <best_model.pt>")

    from modules_hand import HandTracker_onnx

    tracker = HandTracker_onnx().model_hand      # 내부 WilorHandTrackerONNX
    tracker.warmup(np.zeros((PV_HEIGHT, PV_WIDTH, 3), np.uint8))
    runner = ForecastRunner(args.onnx, args.horizons_ms)
    print(f"forecast model = {args.model} ({os.path.relpath(args.onnx)})", flush=True)
    history = HistoryStore(args.history_length, args.max_gap_ms)
    hub = (WebcamSource(args.webcam_index) if args.source == "webcam"
           else HubClient(args.host, args.port,
                          result_kw=KW_HAND_FORECAST, identity=IDENTITY))
    print(f"forecast grid = {args.horizons_ms} ms", flush=True)

    frame_id, sent, latencies = 0, 0, []
    smoother = ForecastSmoother(args.ema_alpha)
    n_saved = 0
    if args.save_vis:
        os.makedirs(args.save_vis, exist_ok=True)
        print(f"시각화 저장: {args.save_vis} ({args.save_every} 프레임마다, "
              f"최대 {args.save_max} 장)", flush=True)
    diag = {}          # track_id -> (직전 anchor, 직전 capture_ns)
    diag_fail = {}     # invalid_reason 별 횟수
    diag_out = {}      # track_id -> 직전 출력 grid (실제 전송값의 떨림을 잰다)
    diag_jitter = []   # (EMA 전 떨림, EMA 후 떨림) mm
    diag_rows = []     # --diag 일 때만 쌓는다
    last_log = (0, 0)  # (시각 ns, frame_id). 로그 구간의 실제 fps 를 재기 위한 것
    poses = []         # --save-pose 용 프레임별 기록
    if args.save_pose:
        print(f"자세 저장: {args.save_pose}", flush=True)
    try:
        while True:
            data = hub.get_latest(timeout=1.0)
            if data is None:
                continue
            received_ns = time.perf_counter_ns()
            loop_start = time.perf_counter_ns()
            packet = proto.HL2SensorPacket()
            packet.ParseFromString(data)
            frame_id += 1

            color = cv2.imdecode(np.frombuffer(packet.image_data, np.uint8), cv2.IMREAD_COLOR)
            if color is None:
                continue
            # depth 도 cam_to_world 도 쓰지 않는다. 자세는 2.5D 로 나가고 절대 위치는
            # 기기가 자기 최신 depth 로 복원한다. HL2 가 depth 를 안 보내도 그대로 돈다.
            resized = cv2.resize(color, (PV_WIDTH, PV_HEIGHT), interpolation=cv2.INTER_AREA)
            # HL2 는 nanosecond timestamp 를 아직 보내지 않는다. legacy float 초를 ns 로
            # 올려 쓰되, 기기 간 절대 시각을 빼지 않는다는 규칙은 그대로 지킨다 (§2.2).
            capture_ns = int(packet.timestamp * 1e9)

            canvas = resized.copy() if (args.vis or args.save_vis) else None
            decode_done = time.perf_counter_ns()
            inference_start = decode_done
            detections = detect_hands(tracker, resized)
            if not detections:
                diag_fail["NO_HAND_DETECTED"] = diag_fail.get("NO_HAND_DETECTED", 0) + 1
            # intrinsics 는 **depth 격자 기준**이다 (HL2 실측: 448x252, fx=349, cx=224).
            # joints_2d 는 640x360 좌표계이므로 그 격자로 옮긴 뒤 fx, cx 를 적용해야 한다.
            # RGB 해상도로 나누면 배율이 1/0.7 배 어긋나고 cx 차이만큼 옆으로 밀린다.
            # depth 를 안 보내더라도 이 두 값은 있어야 intrinsics 를 해석할 수 있다.
            ref_w = packet.depth_width or color.shape[1]
            ref_h = packet.depth_height or color.shape[0]
            scale_x, scale_y = ref_w / PV_WIDTH, ref_h / PV_HEIGHT

            for detection in detections:
                handedness = detection["handedness"]
                track_id = f"{args.device_id}:0:{handedness}"
                reason, grid, blend_weight = "NONE", None, None
                hist_joints = hist_times = None

                anchor = to_25d(detection, (packet.fx, packet.fy, packet.cx, packet.cy),
                                scale_x, scale_y)
                state = history.push(args.device_id, track_id, handedness,
                                     capture_ns, anchor, frame_id)
                if state == "regressed":
                    reason = "TIMESTAMP_REGRESSED"
                elif not history.ready(args.device_id, track_id, handedness):
                    if state == "reset":
                        smoother.reset(track_id)
                    reason = "HISTORY_GAP" if state == "reset" else "INSUFFICIENT_HISTORY"
                else:
                    track = history.get(args.device_id, track_id, handedness)
                    hist_joints, hist_times = track.tensors()
                    grid, blend_weight = blend_with_anchor(
                        runner.run(hist_joints, hist_times, handedness),
                        hist_joints, hist_times,
                        (packet.fx, packet.fy, packet.cx, packet.cy), scale_x, scale_y)

                inference_end = time.perf_counter_ns()
                # 예측 떨림이 기기측 ICP seed 를 흔들지 않도록 EMA 로 다듬는다.
                out_grid = (None if grid is None else smoother(
                    track_id, np.asarray(grid, np.float64)))
                if args.save_pose:
                    nan_h = np.full((history.history_length, NUM_JOINTS, 3), np.nan)
                    nan_g = np.full((len(args.horizons_ms), NUM_JOINTS, 3), np.nan)
                    poses.append({
                        "frame_id": np.float64(frame_id),
                        "capture_ns": np.float64(capture_ns),
                        "is_right": np.float64(handedness == "RIGHT"),
                        "reason": reason,
                        "joints_2d": np.asarray(detection["joints_2d"], np.float64),
                        "root_relative": np.asarray(detection["root_relative"], np.float64),
                        "anchor_25d": anchor,
                        "history_joints": (nan_h if hist_joints is None
                                           else np.asarray(hist_joints, np.float64)),
                        "history_time_ms": (np.full(history.history_length, np.nan)
                                            if hist_times is None
                                            else np.asarray(hist_times, np.float64)),
                        "grid_model": (nan_g if grid is None
                                       else np.asarray(grid, np.float64)),
                        "out_grid": nan_g if out_grid is None else out_grid,
                        "intrinsics": np.array([packet.fx, packet.fy, packet.cx, packet.cy,
                                                scale_x, scale_y], np.float64),
                    })
                    if len(poses) % 300 == 0:        # 중간에 죽어도 남게 한다
                        save_poses(args.save_pose, poses)
                hub.send(fp.build(
                    source_frame_id=frame_id, source_capture_timestamp_ns=capture_ns,
                    legacy_timestamp=packet.timestamp, handedness=handedness,
                    track_id=track_id, horizons_ms=args.horizons_ms,
                    grid=None if out_grid is None else out_grid * SEND_AXES,
                    anchor_25d=anchor * SEND_AXES, model_version=runner.model_version,
                    history_length=args.history_length, valid=grid is not None,
                    invalid_reason=reason,
                    timings={"server_receive_timestamp_ns": received_ns,
                             "inference_start_timestamp_ns": inference_start,
                             "inference_end_timestamp_ns": inference_end,
                             "server_send_timestamp_ns": time.perf_counter_ns()}))
                sent += 1
                latencies.append((inference_end - inference_start) / 1e6)
                if args.diag and reason != "NONE":
                    # 어느 손이 빠졌는지까지 봐야 한다. 화면 밖으로 나간 반대손 때문에
                    # 카운터가 커지는 경우와 주 손이 끊기는 경우는 전혀 다른 문제다.
                    key = f"{reason}/{handedness[0]}"
                    diag_fail[key] = diag_fail.get(key, 0) + 1
                if args.diag and grid is not None:
                    # 실제로 기기에 나가는 값이 프레임마다 얼마나 변하는가.
                    # 사용자가 체감하는 떨림이 이 값이다.
                    raw = np.asarray(grid, np.float64)
                    prev = diag_out.get(track_id)
                    if prev is not None:
                        diag_jitter.append((
                            float(np.linalg.norm(raw - prev[0], axis=-1).mean()),
                            float(np.linalg.norm(out_grid - prev[1], axis=-1).mean())))
                    diag_out[track_id] = (raw, np.asarray(out_grid, np.float64))

                if args.diag:
                    #   step  : anchor 가 프레임마다 얼마나 움직였나 (재투영 px)
                    #   d0    : forecast(h=0) 와 anchor 의 차이
                    #   d_far : 가장 먼 horizon 이 anchor 에서 떨어진 거리
                    px = lambda q: project_to_image(
                        q, (packet.fx, packet.fy, packet.cx, packet.cy), scale_x, scale_y)
                    prev = diag.get(track_id)
                    a_px = px(anchor)
                    step = (np.linalg.norm(a_px - prev[0], axis=-1).mean()
                            if prev else float("nan"))
                    dt_ms = (capture_ns - prev[1]) / 1e6 if prev else float("nan")
                    diag[track_id] = (a_px, capture_ns)
                    if grid is not None:
                        g = np.asarray(grid)
                        d0 = np.linalg.norm(px(g[0]) - a_px, axis=-1).mean()
                        d_far = np.linalg.norm(px(g[-1]) - a_px, axis=-1).mean()
                        diag_rows.append((step, dt_ms, d0, d_far))

                # 그리기는 지연 측정 밖에서 한다
                if canvas is not None:
                    render_forecast(canvas, anchor, out_grid, args.horizons_ms,
                                    args.vis_horizons_ms,
                                    (packet.fx, packet.fy, packet.cx, packet.cy),
                                    scale_x, scale_y, blend_weight)

            if canvas is not None:
                legend = [("tracked", ANCHOR_COLOR)] + [
                    (f"+{int(h)} ms", HORIZON_COLORS[n % len(HORIZON_COLORS)])
                    for n, h in enumerate(args.vis_horizons_ms)]
                for n, (text, color) in enumerate(legend):
                    cv2.putText(canvas, text, (8, 18 + 16 * n), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, color, 1, cv2.LINE_AA)
                if args.save_vis and n_saved < args.save_max and frame_id % args.save_every == 0:
                    cv2.imwrite(os.path.join(args.save_vis, f"f{frame_id:05d}.jpg"), canvas)
                    n_saved += 1
                    if n_saved == args.save_max:
                        print(f"시각화 {n_saved} 장 저장 완료: {args.save_vis}", flush=True)
                if args.vis:
                    cv2.imshow("forecast", canvas)
                    if cv2.waitKey(1) & 0xFF == 27:      # ESC
                        break

            if args.log_every and frame_id % args.log_every == 0:
                recent = np.array(latencies[-args.log_every:]) if latencies else np.zeros(1)
                stats = history.stats()
                if args.diag and diag_fail:
                    print("  fail  " + "  ".join(f"{k}={v}" for k, v in
                          sorted(diag_fail.items(), key=lambda x: -x[1])), flush=True)
                if args.diag:
                    # 실제로 프레임이 들어오는 속도. 모델은 15~30 fps 로 학습했으므로
                    # 이 값이 그 범위를 벗어나면 예측을 그대로 믿으면 안 된다.
                    now = time.perf_counter_ns()
                    fps = ((frame_id - last_log[1]) / ((now - last_log[0]) / 1e9)
                           if last_log[0] else float("nan"))
                    last_log = (now, frame_id)
                    print(f"  rate  {fps:.1f} fps  arrived={hub.n_arrived} "
                          f"dropped={hub.n_dropped} processed={frame_id}  "
                          f"decode={(decode_done - loop_start)/1e6:.1f} ms  "
                          f"loop={(now - loop_start)/1e6:.1f} ms", flush=True)
                if args.diag and diag_rows:
                    a = np.array(diag_rows[-args.log_every:], dtype=float)
                    with np.errstate(invalid="ignore"):
                        print(f"  diag  step={np.nanmedian(a[:, 0]):6.2f} px  "
                              f"dt={np.nanmedian(a[:, 1]):5.1f} ms  "
                              f"|h0-anchor|={np.nanmedian(a[:, 2]):6.2f} px  "
                              f"|h_far-anchor|={np.nanmedian(a[:, 3]):6.2f} px", flush=True)
                        if diag_jitter:
                            j = np.array(diag_jitter[-args.log_every:])
                            print(f"        출력 떨림  EMA 전 {j[:, 0].mean():6.3f}  "
                                  f"EMA 후 {j[:, 1].mean():6.3f}  (2.5D, "
                                  f"alpha={args.ema_alpha})", flush=True)
                print(f"frame {frame_id} sent={sent} "
                      f"inference p50={np.median(recent):.2f} p95={np.percentile(recent, 95):.2f} ms "
                      f"tracks={stats['n_tracks']} ready={stats['n_ready']}", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        print("Cleaning up...")
        if args.save_pose:
            save_poses(args.save_pose, poses)
            print(f"자세 {len(poses)} 프레임 저장: {args.save_pose}", flush=True)
        hub.close()


if __name__ == "__main__":
    main()
