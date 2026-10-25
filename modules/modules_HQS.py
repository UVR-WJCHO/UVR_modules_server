"""HQS — HL2DATA(와 HL2_AUDIO) 로 사용자 상태와 그 변화를 추정한다.

학습한 모델은 쓰지 않는다. 신호 처리와 공용 실시간 얼굴 검출기(OpenCV YuNet)만 쓴다.
특정 장면에 맞춘 규칙 대신 세 층으로 나눈다.

  ① 요소   상황과 무관한 물리·의미 변수를 0~1 소속도로 낸다. 기준값은 장면에 맞춰 고른 것이
           아니라 사람 몸과 행동에서 알려진 범위다 (아래 상수 주석 참고).
             walking      머리의 수평 이동 속도
             seated       선 자세 기준 머리 높이보다 얼마나 낮은가
             head_still   머리가 거의 돌아가지 않는 정도
             face         사람 얼굴이 대화 거리 안에 보이는 정도
             voice        최근 몇 초 중 목소리가 있던 시간의 비율. HL2DATA 에는 소리가 없어(speech 필드는
                          비어 온다) HL2_AUDIO 청크를 webrtcvad 로 20 ms 씩 판정한다
  ② 변화점 BOCPD(Bayesian online change-point detection)로 요소 흐름이 바뀌는 순간을 잡는다.
           상태 정의와 무관하므로 정의하지 않은 상황으로 넘어가도 변화는 잡힌다.
  ③ 상태   요소의 조합으로 선언한다(STATES). 변화점 사이 구간의 평균 소속도로 판정하므로
           프레임 단위로 들락거리지 않는다. 상태 이름이 곧 HL2 에 보내는 모드이고, 어디에도
           들지 않으면 NONE 이다. 새 상태는 STATES 에 한 줄 더한다.

좌표계는 HL2DATA 그대로다(Unity 월드, +Y 위, m). 얼굴 거리는 depth 가 있으면 depth 로, 없으면
얼굴 폭으로 잰다. 내부 파라미터는 수신기의 intrinsics() 가 RGB 해상도로 맞춰 준다.

단독 실행: 수신기의 --record 로 남긴 파일을 다시 돌려 본다.
    python modules/modules_HQS.py --replay output/hqs/session.hl2rec [--gui]
"""
from __future__ import annotations

import math
import os
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YUNET_WEIGHTS = os.path.join(_REPO, "pretrained", "hqs", "face_detection_yunet_2023mar.onnx")

# --- ① 요소 기준값. 전부 사람·행동에서 알려진 범위다 ---------------------------------
WALK_SPEED = 0.3          # m/s. 느린 보행도 0.4 m/s 이상이다. 앉아서 고개를 돌릴 때 머리 이동은 이보다 훨씬 느리다
SPEED_WINDOW_S = 1.0      # 속도를 잴 구간. 한 걸음(약 0.5 s)보다 길게 잡아 상하 흔들림을 평균낸다
SIT_DROP = 0.40           # m. 앉으면 눈높이가 서 있을 때보다 약 0.35~0.45 m 낮아진다
HEAD_TURN_SPEED = 10.0    # °/s. 1 초 동안 머리가 돌아간 각도. 실제 HL2 녹화(s1)에서 머리를 두고 있던
                          # 구간은 2~6 °/s, 고개를 돌리던 구간은 20~100 °/s 였다. 그 사이에 둔다
SOCIAL_DIST = 3.6         # m. 사회적 대화 거리의 바깥 경계 (Hall 의 proxemics)
FACE_WIDTH = 0.15         # m. 성인 얼굴 폭 평균
ATTN_TAU_S = 1.0          # s. 얼굴이 보이는 정도를 순간값이 아니라 1 초 시정수로 본다
DETECT_HZ = 10.0          # 검출기는 이 주기로만 돌린다. 박스는 그 사이 프레임에 그대로 쓴다
VOICE_WINDOW_S = 3.0      # s. 대화는 말과 짧은 쉼(약 0.2 s)이 번갈아 이어진다. 몇 마디가 들어가는 길이로 본다
VOICE_DENSITY = 0.3       # 창 안에서 목소리가 있던 시간 비율. 대화 중엔 화자를 합쳐 보통 이보다 높다
VAD_AGGRESSIVENESS = 2    # webrtcvad 0~3. bridge(와치)와 같은 값
VAD_FRAME_MS = 20

# --- ② 변화점 -------------------------------------------------------------------------
CP_HZ = 5.0               # BOCPD 를 이 주기로 돌린다
CP_EXPECTED_S = 15.0      # 한 구간이 보통 이어지는 시간 (hazard = 1 / 이 값)
CP_MAX_RUN_S = 180.0      # run length 상한
CP_CONFIRM_S = 1.0        # 가장 그럴듯한 run length 가 이만큼 이상 줄면 변화점으로 본다

# --- ③ 상태 ---------------------------------------------------------------------------
# 요소 이름 -> 원하는 값(1 이면 높아야, 0 이면 낮아야). 조건들의 최솟값(퍼지 AND)이 점수다.
# 앞에 둔 상태가 우선이다(general=True 면 순서를 보지 않고 점수로 고른다). FOCUS 는 목소리가 없어야 한다.
# 얼굴이 안 보이는 채로 말하거나 대화가 막 끝났을 때 FOCUS 로 넘어가지 않게 한다
STATES: Dict[str, Dict[str, int]] = {
    "TALKING": {"face": 1, "voice": 1, "walking": 0},
    "MOVING":  {"walking": 1, "seated": 0},
    "FOCUS":   {"seated": 1, "walking": 0, "head_still": 1, "voice": 0},
}
STATE_HOLD_S = {"FOCUS": 5.0}   # 이 시간 동안 조건이 이어져야 들어간다
NONE_STATE = "NONE"
STATE_MIN_SCORE = 0.5     # 이보다 낮으면 그 상태가 아니다
STATE_MIN_AGE_S = 1.0     # 새 구간이 이만큼 지나야 상태를 바꾼다
# 상태 전환에서 생기는 이벤트. ("이전", "다음") -> 이름. "*" 는 아무 상태.
TRANSITION_EVENTS = {("TALKING", "*"): "parted"}

PRIMITIVES = ("walking", "seated", "head_still", "face", "voice")


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
    faces: List[Dict] = field(default_factory=list)
    audio: bool = False                                          # 최근 창에 오디오가 들어왔는지


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

    start_posture: 머리 높이는 상대값뿐이라(원점 = 앱 시작 때 머리) 시작 자세를 알아야 앉음/섬을
                   가를 수 있다. 걷는 것이 보이면 그때의 높이로 선 자세 기준을 다시 잡는다.
    general: True 면 STATES 순서(우선순위)를 보지 않고 성립하는 상태 중 점수가 가장 높은 것을 고른다.
    """

    def __init__(self, start_posture: str = "seated", general: bool = False):
        if start_posture not in ("seated", "standing"):
            raise ValueError("start_posture 는 seated / standing")
        self.start_posture = start_posture
        self._select = self._select_general if general else self._select_ordered
        self.yunet = cv2.FaceDetectorYN_create(YUNET_WEIGHTS, "", (320, 320), 0.7)

        self.head = deque()                      # (t, xyz)
        self.rot = deque()                       # (t, 단위 사원수 xyzw)
        self.y_stand: Optional[float] = None
        self.face = 0.0
        # 오디오는 다른 스레드(수신기의 HL2_AUDIO 루프)에서 들어온다
        import threading
        self._audio_lock = threading.Lock()
        self._vad = None
        self._vad_rate = 0
        self._pcm = np.zeros(0, np.int16)
        self._pcm_t = 0.0
        self.voice_frames = deque()              # (t, 목소리 여부). 20 ms 마다 하나
        self.t_prev: Optional[float] = None
        self.t_det = -1e9
        self.faces: List[Dict] = []

        self.cp = _BOCPD(len(PRIMITIVES), hazard=1.0 / (CP_EXPECTED_S * CP_HZ),
                         max_run=int(CP_MAX_RUN_S * CP_HZ))
        self.t_cp = -1e9
        self.run_prev = 0
        self.recent = deque()                    # (t, vec). 변화가 실제로 일어난 시점부터 구간을 다시 세려고
        self.seg_start: Optional[float] = None
        self.seg_sum = np.zeros(len(PRIMITIVES))
        self.seg_n = 0
        self.held_since: Dict[str, float] = {}   # 상태 -> 점수가 기준을 넘기 시작한 시각
        self.state = NONE_STATE

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

    def _head_still(self, t: float, q: np.ndarray) -> float:
        """1 초 창 양 끝 머리 자세 사이의 회전각 / 시간. 프레임마다 더하면 추적 잡음이 쌓인다."""
        q = q / np.linalg.norm(q)
        self.rot.append((t, q))
        while self.rot and t - self.rot[0][0] > SPEED_WINDOW_S:
            self.rot.popleft()
        t0, q0 = self.rot[0]
        dt = t - t0
        if dt <= 0.3:
            return 1.0
        turn = math.degrees(2.0 * math.acos(min(1.0, abs(float(q @ q0))))) / dt
        return 1.0 - _member(turn, HEAD_TURN_SPEED, 3.0)

    def _detect(self, rgb: np.ndarray, K: np.ndarray, depth=None) -> None:
        h, w = rgb.shape[:2]
        self.yunet.setInputSize((w, h))
        _, faces = self.yunet.detect(rgb)
        self.faces = []
        for f in (faces if faces is not None else []):
            x, y, fw, fh = f[:4]
            box = [float(x), float(y), float(x + fw), float(y + fh)]
            dist = _box_depth_m(depth, box, w, h, min_cover=0.2)
            if dist is None:
                dist = float(K[0, 0]) * FACE_WIDTH / max(1.0, float(fw))
            self.faces.append({"box": box, "dist": dist})

    # ---------------------------------------------------------------- ③ 상태 고르기
    def _qualified(self, scores: Dict[str, float], t: float) -> List[str]:
        """구간 평균 점수가 기준을 넘고, 지속 시간이 정해진 상태는 그만큼 끊기지 않은 상태들."""
        return [s for s in STATES if scores[s] >= STATE_MIN_SCORE and (
                s == self.state or t - self.held_since.get(s, t) >= STATE_HOLD_S.get(s, 0.0))]

    def _select_ordered(self, scores: Dict[str, float], t: float) -> str:
        """STATES 에 적은 순서가 우선순위다. 성립하는 첫 상태."""
        q = self._qualified(scores, t)
        return q[0] if q else NONE_STATE

    def _select_general(self, scores: Dict[str, float], t: float) -> str:
        """순서를 보지 않는다. 성립하는 상태 중 구간 평균 점수가 가장 높은 것."""
        q = self._qualified(scores, t)
        return max(q, key=scores.get) if q else NONE_STATE

    # ---------------------------------------------------------------- 오디오
    def update_audio(self, pcm: bytes, sample_rate: int, t_start: float, channels: int = 1) -> None:
        """HL2_AUDIO 청크 하나(int16 LE). 20 ms 씩 끊어 목소리 여부를 시각과 함께 남긴다.

        t_start 는 청크 첫 샘플의 시각(초)으로 HL2DATA timestamp 와 같은 축이다. 청크(32 ms)가
        20 ms 로 나눠떨어지지 않으므로 남은 샘플은 다음 청크와 이어 붙인다.
        """
        from devicebridge.vad import VadProcessor
        x = np.frombuffer(pcm, "<i2")
        if channels > 1:
            x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1).astype(np.int16)
        rate = int(sample_rate)
        if rate > 32000 and rate % 16000 == 0:              # 기기가 16 kHz 를 못 열면 원래 레이트로 온다
            k = rate // 16000
            x = x[: len(x) // k * k].reshape(-1, k).mean(axis=1).astype(np.int16)
            rate = 16000
        with self._audio_lock:
            if self._vad is None or rate != self._vad_rate:
                self._vad, self._vad_rate = VadProcessor(VAD_AGGRESSIVENESS, rate, VAD_FRAME_MS), rate
                self._pcm = np.zeros(0, np.int16)
            if self._pcm.size == 0 or abs(t_start - self._pcm_t - self._pcm.size / rate) > 0.1:
                self._pcm, self._pcm_t = np.zeros(0, np.int16), t_start   # 끊겼으면 새로 시작
            self._pcm = np.concatenate([self._pcm, x])
            n = rate * VAD_FRAME_MS // 1000
            while self._pcm.size >= n:
                self.voice_frames.append((self._pcm_t, bool(self._vad.is_speech(self._pcm[:n].tobytes()))))
                self._pcm, self._pcm_t = self._pcm[n:], self._pcm_t + VAD_FRAME_MS / 1000.0
            while self.voice_frames and self._pcm_t - self.voice_frames[0][0] > 2 * VOICE_WINDOW_S:
                self.voice_frames.popleft()

    def _voice(self, t: float):
        with self._audio_lock:
            v = [s for ts, s in self.voice_frames if t - VOICE_WINDOW_S <= ts <= t]
        if not v:
            return 0.0, False
        return _member(float(np.mean(v)), VOICE_DENSITY, 0.08), True

    # ---------------------------------------------------------------- 매 패킷
    def update(self, pkt, rgb: Optional[np.ndarray], K: np.ndarray,
               depth: Optional[np.ndarray] = None) -> HQSResult:
        t = float(pkt.timestamp)
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        self.t_prev = t
        prim = self._locomotion(t, np.array([pkt.head_pos_x, pkt.head_pos_y, pkt.head_pos_z], float))
        prim["head_still"] = self._head_still(
            t, np.array([pkt.head_rot_x, pkt.head_rot_y, pkt.head_rot_z, pkt.head_rot_w], float))

        if rgb is not None:
            if t - self.t_det >= 1.0 / DETECT_HZ:
                self._detect(rgb, K, depth)
                self.t_det = t
            near = max((_member(SOCIAL_DIST - f["dist"], 0.0, 0.4) for f in self.faces), default=0.0)
            a = 1.0 - math.exp(-dt / ATTN_TAU_S) if dt > 0 else 1.0
            self.face += a * (near - self.face)             # 순간값이 아니라 1 초 시정수로 머문 정도
        prim["face"] = self.face
        prim["voice"], audio_ok = self._voice(t)
        vec = np.array([prim[k] for k in PRIMITIVES])

        # ② 변화점
        change = False
        if self.seg_start is None:
            self.seg_start = t
        self.recent.append((t, vec))
        while t - self.recent[0][0] > CP_MAX_RUN_S:
            self.recent.popleft()
        if t - self.t_cp >= 1.0 / CP_HZ:
            self.t_cp = t
            run = int(np.argmax(self.cp.update(vec)))
            # 가장 그럴듯한 run length 는 매 단계 하나씩 자란다. 크게 줄었으면 더 최근에 시작한 구간을 택한 것이다.
            # 짧은 run 쪽 확률 합으로 보면 놓친다. 얼굴·목소리는 몇 초에 걸쳐 바뀌어서 알아챌 때 run 이 이미 길다
            if run <= self.run_prev - CP_CONFIRM_S * CP_HZ:
                change = True
                t_change = t - run / CP_HZ                   # 그 run 만큼 거슬러 올라가 구간을 시작한다
                kept = [v for tt, v in self.recent if tt >= t_change]
                self.seg_start = t_change
                self.seg_sum = np.sum(kept, axis=0) if kept else np.zeros(len(PRIMITIVES))
                self.seg_n = len(kept)
            self.run_prev = run
        if not change:
            self.seg_sum += vec
            self.seg_n += 1

        # ③ 상태 (구간 평균으로). 지속 시간은 구간 평균이 아니라 매 순간 조건이 끊기지 않았는지로 잰다.
        # 평균은 '가만히 4 초, 둘러보기 3 초' 의 반복도 기준 위로 올려 버린다
        mean = dict(zip(PRIMITIVES, self.seg_sum / max(1, self.seg_n)))
        score = lambda v: {s: min(v[p] if want else 1.0 - v[p] for p, want in cond.items())
                           for s, cond in STATES.items()}
        scores = score(mean)
        for s, v in score(prim).items():
            if v < STATE_MIN_SCORE:
                self.held_since.pop(s, None)
            else:
                self.held_since.setdefault(s, t)
        label = self._select(scores, t)
        events: List[str] = []
        if label != self.state and t - self.seg_start >= STATE_MIN_AGE_S:
            for (a_, b_), name in TRANSITION_EVENTS.items():
                if a_ in (self.state, "*") and b_ in (label, "*") and self.state != label:
                    events.append(name)
            events.insert(0, f"{self.state}->{label}")
            self.state = label

        prim.pop("_speed", None)
        return HQSResult(t=t, primitives=prim, state=self.state, scores=scores, change_point=change,
                         events=events, faces=list(self.faces), audio=audio_ok)


def _box_depth_m(depth, box, w, h, min_cover: float = 0.1) -> Optional[float]:
    """박스 안 유효 depth 의 중앙값(m). 유효 화소가 min_cover 보다 적으면 None."""
    if depth is None:
        return None
    dh, dw = depth.shape[:2]
    x1, y1, x2, y2 = box
    xa, xb = max(0, int(x1 * dw / w)), min(dw, int(np.ceil(x2 * dw / w)))
    ya, yb = max(0, int(y1 * dh / h)), min(dh, int(np.ceil(y2 * dh / h)))
    patch = depth[ya:yb, xa:xb]
    if patch.size == 0:
        return None
    v = patch[patch > 0]
    if v.size < min_cover * patch.size:
        return None
    return float(np.median(v)) / 1000.0


def draw(img: np.ndarray, r: HQSResult) -> None:
    """수신기 overlay 위에 상태, 요소 막대, 얼굴 박스를 그린다."""
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
        label = "voice(no mic)" if k == "voice" and not r.audio else k
        cv2.putText(img, label, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------------------------------ 녹화 형식
# 수신기 --record: [8 byte 수신 시각(double, big-endian)][4 byte 길이][HL2SensorPacket bytes] 반복.
# 오디오는 같은 형식으로 <파일>.audio 에 HL2Audio bytes 를 따로 쓴다(HL2DATA 녹화 형식은 그대로 둔다).
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
    import hl2_sensors_pb2 as sensproto
    from main_all_hl2_receiver import decode_depth, decode_rgb, intrinsics

    ap = argparse.ArgumentParser(description="녹화한 HL2DATA(+ .audio) 를 HQS 로 다시 돌려 본다")
    ap.add_argument("--replay", required=True, help="수신기 --record 로 남긴 파일")
    ap.add_argument("--start-posture", default="seated", choices=("seated", "standing"))
    ap.add_argument("--general", action="store_true", help="상태 우선순위 대신 점수가 가장 높은 상태를 고른다")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    hqs = HQS(start_posture=args.start_posture, general=args.general)
    audio = []                                               # (청크 시작 시각 s, HL2Audio)
    if os.path.exists(args.replay + ".audio"):
        for _t, d in read_recording(args.replay + ".audio"):
            a = sensproto.HL2Audio()
            a.ParseFromString(d)
            audio.append((a.start_timestamp / 1e9, a))
        audio.sort(key=lambda x: x[0])
    print(f"오디오 청크 {len(audio)}개" + ("" if audio else " (없음: voice 는 0 으로 나온다)"), flush=True)
    ai, n = 0, 0
    for _t_recv, data in read_recording(args.replay):
        pkt = proto.HL2SensorPacket()
        pkt.ParseFromString(data)
        while ai < len(audio) and audio[ai][0] <= pkt.timestamp:   # 이 프레임 시각까지의 소리를 먼저 넣는다
            a = audio[ai][1]
            hqs.update_audio(a.audio, a.sample_rate, audio[ai][0], max(1, a.channels))
            ai += 1
        rgb, depth = decode_rgb(pkt.image_data), decode_depth(pkt.depth_data)
        h, w = (rgb.shape[:2] if rgb is not None else (pkt.depth_height or 360, pkt.depth_width or 640))
        r = hqs.update(pkt, rgb, intrinsics(pkt, w, h), depth)
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
