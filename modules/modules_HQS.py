"""HQS — HL2DATA 패킷만으로 사용자 상태와 그 변화를 추정한다.

학습한 모델은 쓰지 않는다. 신호 처리와 공용 실시간 검출기(COCO YOLO, OpenCV YuNet)만 쓴다.
특정 장면에 맞춘 규칙 대신 세 층으로 나눈다.

  ① 요소   상황과 무관한 물리·의미 변수를 0~1 소속도로 낸다. 기준값은 장면에 맞춰 고른 것이
           아니라 사람 몸과 행동에서 알려진 범위다 (아래 상수 주석 참고).
             walking      머리의 수평 이동 속도
             seated       선 자세 기준 머리 높이보다 얼마나 낮은가
             attn_screen  시선이 화면(tv, laptop) 위에 머무는 정도
             attn_face    시선이 실제 사람 얼굴 위에 머무는 정도
             engaged      가까이에서 이쪽을 정면으로 보는 얼굴이 시야 가운데 있는 정도
  ② 변화점 BOCPD(Bayesian online change-point detection)로 요소 흐름이 바뀌는 순간을 잡는다.
           상태 정의와 무관하므로 정의하지 않은 상황으로 넘어가도 변화는 잡힌다.
  ③ 상태   요소의 조합으로 선언한다(STATES). 변화점 사이 구간의 평균 소속도로 판정하므로
           프레임 단위로 들락거리지 않고, 별도 타이머도 없다. 새 상태는 STATES 에 한 줄 더한다.

좌표계는 HL2DATA 그대로다. 머리·시선·손은 Unity 월드(+Y 위, m), 투영은 수신기의
project_points(카메라 -Z 전방)를 주입받아 쓴다. fx 등 내부 파라미터는 패킷 값이 정확하다고 본다.

단독 실행: 수신기의 --record 로 남긴 파일을 다시 돌려 본다.
    python modules/modules_HQS.py --replay output/hqs/session.hl2rec [--gui]
"""
from __future__ import annotations

import math
import os
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_WEIGHTS = os.path.join(_REPO, "pretrained", "object", "yolo11n.pt")          # COCO 80 클래스
YUNET_WEIGHTS = os.path.join(_REPO, "pretrained", "hqs", "face_detection_yunet_2023mar.onnx")

# --- ① 요소 기준값. 전부 사람·행동에서 알려진 범위다 ---------------------------------
WALK_SPEED = 0.3          # m/s. 느린 보행도 0.4 m/s 이상이다. 앉아서 고개를 돌릴 때 머리 이동은 이보다 훨씬 느리다
SPEED_WINDOW_S = 1.0      # 속도를 잴 구간. 한 걸음(약 0.5 s)보다 길게 잡아 상하 흔들림을 평균낸다
SIT_DROP = 0.40           # m. 앉으면 눈높이가 서 있을 때보다 약 0.35~0.45 m 낮아진다
SOCIAL_DIST = 3.6         # m. 사회적 대화 거리의 바깥 경계 (Hall 의 proxemics)
FACE_WIDTH = 0.15         # m. 성인 얼굴 폭 평균
CENTER_ANGLE = 30.0       # deg. 상대를 향하고 있다고 볼 수평 각도
FRONTAL_OFFSET = 0.25     # 코가 두 눈 중점에서 눈 사이 거리의 이 비율 안이면 정면
ATTN_TAU_S = 1.0          # s. 시선이 머무는 '정도' 를 순간값이 아니라 1 초 시정수로 본다
GAZE_DIST_M = 1.5         # depth 가 없을 때 시선 끝점을 둘 거리
BOX_MARGIN = 0.15         # 시선 정밀도(1~2°)만큼 박스를 넓혀 판정
DETECT_HZ = 10.0          # 검출기는 이 주기로만 돌린다. 박스는 그 사이 프레임에 그대로 쓴다

SCREEN_CLASSES = ("tv", "laptop")

# --- ② 변화점 -------------------------------------------------------------------------
CP_HZ = 5.0               # BOCPD 를 이 주기로 돌린다
CP_EXPECTED_S = 15.0      # 한 구간이 보통 이어지는 시간 (hazard = 1 / 이 값)
CP_MAX_RUN_S = 180.0      # run length 상한
CP_CONFIRM_S = 1.0        # run length 가 이보다 짧을 확률이 절반을 넘으면 변화점으로 본다

# --- ③ 상태 ---------------------------------------------------------------------------
# 요소 이름 -> 원하는 값(1 이면 높아야, 0 이면 낮아야). 조건들의 최솟값(퍼지 AND)이 점수다.
STATES: Dict[str, Dict[str, int]] = {
    "focus_seated": {"seated": 1, "walking": 0, "attn_screen": 1},
    "walking":      {"walking": 1, "seated": 0},
    "conversation": {"engaged": 1, "walking": 0, "attn_face": 1},
}
STATE_MIN_SCORE = 0.5     # 이보다 낮으면 어느 상태에도 넣지 않는다 (unknown)
STATE_MIN_AGE_S = 1.0     # 새 구간이 이만큼 지나야 상태를 바꾼다
# 상태 전환에서 생기는 이벤트. ("이전", "다음") -> 이름. "*" 는 아무 상태.
TRANSITION_EVENTS = {("conversation", "*"): "parted"}

PRIMITIVES = ("walking", "seated", "attn_screen", "attn_face", "engaged")


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def _member(value: float, center: float, width: float) -> float:
    """value 가 center 보다 크면 1 에 가까워지는 소속도. width 는 경계의 부드러움."""
    return _sig((value - center) / width)


@dataclass
class HQSResult:
    t: float
    primitives: Dict[str, float]
    state: str
    scores: Dict[str, float]
    change_point: bool
    events: List[str] = field(default_factory=list)
    screens: List[List[float]] = field(default_factory=list)     # 그리기용 박스 xyxy
    faces: List[Dict] = field(default_factory=list)
    gaze_uv: Optional[List[float]] = None


class _BOCPD:
    """Adams & MacKay (2007). 차원마다 정규-감마 사전분포를 둔 독립 가우시안. 학습하지 않는다."""

    def __init__(self, dim: int, hazard: float, max_run: int,
                 mu0: float = 0.5, kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 0.01):
        self.h, self.max_run = hazard, max_run
        self.p0 = (mu0, kappa0, alpha0, beta0)
        self.dim = dim
        self.R = np.array([1.0])
        self.mu = np.full((1, dim), mu0)
        self.kappa = np.full((1, 1), kappa0)
        self.alpha = np.full((1, 1), alpha0)
        self.beta = np.full((1, dim), beta0)

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)[None, :]
        # 각 run length 가정에서의 Student-t 예측 로그우도 (차원 독립)
        df = 2 * self.alpha
        scale2 = self.beta * (self.kappa + 1) / (self.alpha * self.kappa)
        z = (x - self.mu) ** 2 / (df * scale2)
        logp = (_lgamma((df + 1) / 2) - _lgamma(df / 2) - 0.5 * np.log(np.pi * df * scale2)
                - (df + 1) / 2 * np.log1p(z)).sum(axis=1)
        logp -= logp.max()
        pred = np.exp(logp)
        growth = self.R * pred * (1 - self.h)
        cp = (self.R * pred * self.h).sum()
        R = np.append(cp, growth)
        R /= R.sum()
        # 충분통계 갱신 (run length 0 은 사전분포에서 다시 시작)
        mu0, kappa0, alpha0, beta0 = self.p0
        mu_new = (self.kappa * self.mu + x) / (self.kappa + 1)
        beta_new = self.beta + self.kappa * (x - self.mu) ** 2 / (2 * (self.kappa + 1))
        self.mu = np.vstack([np.full((1, self.dim), mu0), mu_new])
        self.beta = np.vstack([np.full((1, self.dim), beta0), beta_new])
        self.kappa = np.vstack([[[kappa0]], self.kappa + 1])
        self.alpha = np.vstack([[[alpha0]], self.alpha + 0.5])
        if len(R) > self.max_run:
            R, self.mu, self.beta = R[:self.max_run], self.mu[:self.max_run], self.beta[:self.max_run]
            self.kappa, self.alpha = self.kappa[:self.max_run], self.alpha[:self.max_run]
            R /= R.sum()
        self.R = R
        return R


def _lgamma(a: np.ndarray) -> np.ndarray:
    return np.vectorize(math.lgamma)(a)


class HQS:
    """패킷마다 update() 를 부르면 HQSResult 를 돌려준다.

    project: 수신기의 project_points(K, cam_to_world, P) -> (uv, valid). 투영 규약을 한 곳에 둔다.
    start_posture: 머리 높이는 상대값뿐이라(원점 = 앱 시작 때 머리) 시작 자세를 알아야 앉음/섬을
                   가를 수 있다. 걷는 것이 보이면 그때의 높이로 선 자세 기준을 다시 잡는다.
    """

    def __init__(self, project: Callable, start_posture: str = "seated", device: Optional[str] = None):
        if start_posture not in ("seated", "standing"):
            raise ValueError("start_posture 는 seated / standing")
        from ultralytics import YOLO
        self.project = project
        self.start_posture = start_posture
        self.yolo = YOLO(YOLO_WEIGHTS)
        self.device = device if device is not None else (0 if _cuda() else "cpu")
        names = self.yolo.names
        self.screen_ids = [k for k, v in names.items() if v in SCREEN_CLASSES]
        self.yunet = cv2.FaceDetectorYN_create(YUNET_WEIGHTS, "", (320, 320), 0.7)

        self.head = deque()                      # (t, xyz)
        self.y_stand: Optional[float] = None
        self.attn = {"attn_screen": 0.0, "attn_face": 0.0, "engaged": 0.0}
        self.t_prev: Optional[float] = None
        self.t_det = -1e9
        self.screens: List[List[float]] = []
        self.faces: List[Dict] = []

        self.cp = _BOCPD(len(PRIMITIVES), hazard=1.0 / (CP_EXPECTED_S * CP_HZ),
                         max_run=int(CP_MAX_RUN_S * CP_HZ))
        self.t_cp = -1e9
        self.cp_steps_since = 0
        self.recent = deque()                    # (t, vec). 변화가 실제로 일어난 시점부터 구간을 다시 세려고
        self.seg_start: Optional[float] = None
        self.seg_sum = np.zeros(len(PRIMITIVES))
        self.seg_n = 0
        self.state = "unknown"

    # ---------------------------------------------------------------- ① 요소
    def _locomotion(self, t: float, p: np.ndarray) -> Dict[str, float]:
        self.head.append((t, p))
        while self.head and t - self.head[0][0] > SPEED_WINDOW_S:
            self.head.popleft()
        t0, p0 = self.head[0]
        dt = t - t0
        speed = float(np.hypot(p[0] - p0[0], p[2] - p0[2]) / dt) if dt > 0.3 else 0.0   # 수평(x, z)
        walking = _member(speed, WALK_SPEED, 0.07)

        y = float(p[1])
        if self.y_stand is None:
            self.y_stand = y + (SIT_DROP if self.start_posture == "seated" else 0.0)
        if walking > 0.8:                                    # 걷고 있으면 서 있다. 그 높이로 기준을 맞춘다
            self.y_stand += 0.05 * (y - self.y_stand)
        seated = _member(self.y_stand - y, SIT_DROP / 2, 0.05)
        return {"walking": walking, "seated": seated, "_speed": speed}

    def _detect(self, rgb: np.ndarray) -> None:
        r = self.yolo.predict(rgb, verbose=False, device=self.device, conf=0.35, classes=self.screen_ids)[0]
        self.screens = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
        h, w = rgb.shape[:2]
        self.yunet.setInputSize((w, h))
        _, faces = self.yunet.detect(rgb)
        self.faces = []
        for f in (faces if faces is not None else []):
            x, y, fw, fh = f[:4]
            lm = f[4:14].reshape(5, 2)                       # 오른눈, 왼눈, 코, 입 오른쪽, 입 왼쪽
            cx, cy = x + fw / 2, y + fh / 2
            # 화면 안에 있는 얼굴은 모니터 속 사람이다. 실제 사람으로 세지 않는다
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in self.screens):
                continue
            self.faces.append({"box": [float(x), float(y), float(x + fw), float(y + fh)], "lm": lm})

    def _social(self, K: np.ndarray, gaze_uv: Optional[np.ndarray]) -> Dict[str, float]:
        fx, cx = float(K[0, 0]), float(K[0, 2])
        engaged, on_face = 0.0, 0.0
        for f in self.faces:
            x1, y1, x2, y2 = f["box"]
            dist = fx * FACE_WIDTH / max(1.0, x2 - x1)
            near = _member(SOCIAL_DIST - dist, 0.0, 0.4)
            angle = math.degrees(math.atan2(abs((x1 + x2) / 2 - cx), fx))
            centered = _member(CENTER_ANGLE - angle, 0.0, 5.0)
            lm = f["lm"]
            eye_mid, eye_d = (lm[0] + lm[1]) / 2, max(1.0, float(np.linalg.norm(lm[0] - lm[1])))
            frontal = _member(FRONTAL_OFFSET - abs(lm[2][0] - eye_mid[0]) / eye_d, 0.0, 0.05)
            f["dist"], f["frontal"] = dist, frontal
            engaged = max(engaged, min(near, centered, frontal))
            if gaze_uv is not None and _inside(gaze_uv, f["box"]):
                on_face = 1.0
        on_screen = 1.0 if gaze_uv is not None and any(_inside(gaze_uv, b) for b in self.screens) else 0.0
        return {"attn_screen": on_screen, "attn_face": on_face, "engaged": engaged}

    def _gaze_uv(self, pkt, K, c2w, depth) -> Optional[np.ndarray]:
        if c2w is None:
            return None
        go = np.array([pkt.eye_gaze_origin_x, pkt.eye_gaze_origin_y, pkt.eye_gaze_origin_z], np.float32)
        gd = np.array([pkt.eye_gaze_dir_x, pkt.eye_gaze_dir_y, pkt.eye_gaze_dir_z], np.float32)
        n = float(np.linalg.norm(gd))
        if n < 1e-3:
            return None
        gd /= n
        d = GAZE_DIST_M
        for _ in range(2):                                   # depth 가 있으면 그 지점 거리로 한 번 고친다
            uv, ok = self.project(K, c2w, (go + d * gd).reshape(1, 3))
            if not ok[0]:
                return None
            u, v = uv[0]
            if depth is None:
                break
            yi, xi = int(round(v)), int(round(u))
            if not (0 <= yi < depth.shape[0] and 0 <= xi < depth.shape[1]) or depth[yi, xi] <= 0:
                break
            d = float(depth[yi, xi]) / 1000.0
        return np.array([u, v])

    # ---------------------------------------------------------------- 매 패킷
    def update(self, pkt, rgb: Optional[np.ndarray], K: np.ndarray, c2w: Optional[np.ndarray],
               depth: Optional[np.ndarray] = None) -> HQSResult:
        t = float(pkt.timestamp)
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        self.t_prev = t
        prim = self._locomotion(t, np.array([pkt.head_pos_x, pkt.head_pos_y, pkt.head_pos_z], float))

        gaze_uv = None
        if rgb is not None:
            if t - self.t_det >= 1.0 / DETECT_HZ:
                self._detect(rgb)
                self.t_det = t
            gaze_uv = self._gaze_uv(pkt, K, c2w, depth)
            obs = self._social(K, gaze_uv)
            a = 1.0 - math.exp(-dt / ATTN_TAU_S) if dt > 0 else 1.0
            for k, v in obs.items():                         # 순간값이 아니라 1 초 시정수로 머문 정도
                self.attn[k] += a * (v - self.attn[k])
        prim.update(self.attn)
        vec = np.array([prim[k] for k in PRIMITIVES])

        # ② 변화점
        change = False
        if self.seg_start is None:
            self.seg_start = t
        self.recent.append((t, vec))
        while t - self.recent[0][0] > 2 * CP_CONFIRM_S + 1.0:
            self.recent.popleft()
        if t - self.t_cp >= 1.0 / CP_HZ:
            self.t_cp = t
            R = self.cp.update(vec)
            k = max(1, int(CP_CONFIRM_S * CP_HZ))
            self.cp_steps_since += 1
            if R[:k].sum() > 0.5 and self.cp_steps_since > k:
                change = True
                self.cp_steps_since = 0
                # 감지는 몇 단계 늦다. 가장 그럴듯한 run length 만큼 거슬러 올라가 구간을 시작한다
                t_change = t - int(np.argmax(R[:k])) / CP_HZ
                kept = [v for tt, v in self.recent if tt >= t_change]
                self.seg_start = t_change
                self.seg_sum = np.sum(kept, axis=0) if kept else np.zeros(len(PRIMITIVES))
                self.seg_n = len(kept)
        if not change:
            self.seg_sum += vec
            self.seg_n += 1

        # ③ 상태 (구간 평균으로)
        mean = dict(zip(PRIMITIVES, self.seg_sum / max(1, self.seg_n)))
        scores = {s: min(mean[p] if want else 1.0 - mean[p] for p, want in cond.items())
                  for s, cond in STATES.items()}
        best = max(scores, key=scores.get)
        label = best if scores[best] >= STATE_MIN_SCORE else "unknown"
        events: List[str] = []
        if label != self.state and t - self.seg_start >= STATE_MIN_AGE_S:
            for (a_, b_), name in TRANSITION_EVENTS.items():
                if a_ in (self.state, "*") and b_ in (label, "*") and self.state != label:
                    events.append(name)
            events.insert(0, f"{self.state}->{label}")
            self.state = label

        prim.pop("_speed", None)
        return HQSResult(t=t, primitives=prim, state=self.state, scores=scores, change_point=change,
                         events=events, screens=list(self.screens), faces=list(self.faces),
                         gaze_uv=None if gaze_uv is None else gaze_uv.tolist())


def _inside(uv, box) -> bool:
    x1, y1, x2, y2 = box
    mx, my = BOX_MARGIN * (x2 - x1), BOX_MARGIN * (y2 - y1)
    return x1 - mx <= uv[0] <= x2 + mx and y1 - my <= uv[1] <= y2 + my


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def draw(img: np.ndarray, r: HQSResult) -> None:
    """수신기 overlay 위에 상태, 요소 막대, 화면·얼굴 박스를 그린다."""
    for b in r.screens:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 160, 0), 2)
    for f in r.faces:
        b = f["box"]
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 255), 2)
        cv2.putText(img, f"{f.get('dist', 0):.1f}m", (int(b[0]), int(b[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    h, w = img.shape[:2]
    x0, y0 = w - 210, 20
    cv2.rectangle(img, (x0 - 8, 4), (w - 4, y0 + 18 * (len(r.primitives) + 1) + 4), (0, 0, 0), -1)
    cv2.putText(img, f"STATE {r.state}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    for i, (k, v) in enumerate(r.primitives.items()):
        y = y0 + 18 * (i + 1)
        cv2.rectangle(img, (x0 + 95, y - 10), (x0 + 95 + int(100 * v), y - 2), (0, 200, 0), -1)
        cv2.putText(img, k, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------------------------------ 녹화 형식
# 수신기 --record: [8 byte 수신 시각(double, big-endian)][4 byte 길이][HL2SensorPacket bytes] 반복
REC_HEADER = struct.Struct(">dI")


def read_recording(path: str):
    with open(path, "rb") as f:
        while True:
            head = f.read(REC_HEADER.size)
            if len(head) < REC_HEADER.size:
                return
            t_recv, n = REC_HEADER.unpack(head)
            data = f.read(n)
            if len(data) < n:
                return
            yield t_recv, data


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, _REPO)
    sys.path.insert(0, os.path.join(_REPO, "_comm"))
    import hl2_data_pb2 as proto
    from main_all_hl2_receiver import decode_depth, decode_rgb, intrinsics, pose_from_packet, project_points

    ap = argparse.ArgumentParser(description="녹화한 HL2DATA 를 HQS 로 다시 돌려 본다")
    ap.add_argument("--replay", required=True, help="수신기 --record 로 남긴 파일")
    ap.add_argument("--start-posture", default="seated", choices=("seated", "standing"))
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    hqs = HQS(project_points, start_posture=args.start_posture)
    n = 0
    for _t_recv, data in read_recording(args.replay):
        pkt = proto.HL2SensorPacket()
        pkt.ParseFromString(data)
        rgb, depth = decode_rgb(pkt.image_data), decode_depth(pkt.depth_data)
        h, w = (rgb.shape[:2] if rgb is not None else (pkt.depth_height or 360, pkt.depth_width or 640))
        c2w, _ = pose_from_packet(pkt)
        r = hqs.update(pkt, rgb, intrinsics(pkt, w, h), c2w, depth)
        if r.change_point:
            print(f"t={r.t:8.2f}  change point", flush=True)
        for e in r.events:
            print(f"t={r.t:8.2f}  {e}", flush=True)
        if args.gui and rgb is not None:
            vis = rgb.copy()
            draw(vis, r)
            cv2.imshow("hqs_replay", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        n += 1
    print(f"{n} packets, final state {hqs.state}")
