"""실시간(스트리밍) 지표 계산 — 라이브 녹화 중 패널 표시용.

각 스트림 스레드가 push_*()로 데이터를 넣고, 메인 루프가 current(now)로 최근 W초
causal 윈도우의 지표값을 얻는다. 오프라인 배치(extract_metrics)와 동일한 수식.
버퍼는 deque, 접근은 lock으로 보호(스레드 안전).
"""

import threading
from collections import deque
import numpy as np

from . import common as C
from . import haptic


class StreamingMetrics:
    def __init__(self, window=2.0, accel_has_gravity=True):
        """accel_has_gravity: push_accel 로 들어오는 값에 중력이 섞여 있는가.

        가속도계 원본(hl2ss RM_IMU)은 True 다. 정지 상태에서도 1 g 가 나오므로
        moving average 로 저주파를 빼야 선형가속도가 남는다.

        head pose 를 미분한 값은 False 다. 기구학적 미분이라 중력이 애초에 없고,
        여기서 저주파를 빼면 실제 저주파 운동을 지워 지표가 조용히 틀린다.
        """
        self.W = window
        self.accel_has_gravity = accel_has_gravity
        self._lock = threading.Lock()
        self._acc = deque()   # (ts, xyz[N,3])  per packet
        self._gyr = deque()   # (ts, xyz[N,3])
        self._hand = {'L': deque(), 'R': deque()}  # (ts, x, y, z, valid) per sample
        self._vis = deque()   # (ts, clutter)
        self._flow = deque()  # (ts, residual flow px/s)
        self._aud = deque()   # (ts, density)
        self._gaze = deque()  # (ts, yaw, pitch)  head 기준 deg
        self._latest = None   # 가장 최근에 들어온 timestamp. '지금' 의 기준이다

    @property
    def latest_timestamp(self):
        """가장 최근 샘플의 시각. `current()` 에 넘길 '지금' 이다.

        벽시계를 쓰면 안 된다. 패킷 timestamp 의 원점은 앱 기동(또는 기기 부팅)이라
        Unix epoch 와 축이 다르고, 그대로 빼면 창 밖으로 전부 밀려 모든 지표가 NaN 이 된다.
        """
        with self._lock:
            return self._latest

    def _mark(self, ts):
        if self._latest is None or ts > self._latest:
            self._latest = ts

    def _trim(self, dq, now):
        lo = now - (self.W + 0.5) * C.QPC
        while dq and dq[0][0] < lo:
            dq.popleft()

    # ---- push (스트림 스레드에서 호출) ----
    def push_accel(self, ts, xyz):
        with self._lock:
            self._acc.append((ts, np.asarray(xyz, np.float32)))
            self._trim(self._acc, ts)
            self._mark(ts)

    def push_gyro(self, ts, xyz):
        with self._lock:
            self._gyr.append((ts, np.asarray(xyz, np.float32)))
            self._trim(self._gyr, ts)
            self._mark(ts)

    def push_hand(self, side, ts, pos, valid):
        with self._lock:
            self._hand[side].append((ts, float(pos[0]), float(pos[1]), float(pos[2]), bool(valid)))
            self._trim(self._hand[side], ts)
            self._mark(ts)

    def push_clutter(self, ts, clutter):
        with self._lock:
            self._vis.append((ts, float(clutter)))
            self._trim(self._vis, ts)
            self._mark(ts)

    def push_flow(self, ts, flow):
        with self._lock:
            self._flow.append((ts, float(flow)))
            self._trim(self._flow, ts)
            self._mark(ts)

    def push_audio(self, ts, density):
        with self._lock:
            self._aud.append((ts, float(density)))
            self._trim(self._aud, ts)
            self._mark(ts)

    def push_gaze(self, ts, yaw, pitch):
        with self._lock:
            self._gaze.append((ts, float(yaw), float(pitch)))
            self._trim(self._gaze, ts)
            self._mark(ts)

    # ---- read (메인 루프에서 호출) ----
    def occupancy(self, now):
        """각 지표가 창 안의 샘플 몇 개로 계산됐는지.

        값이 나온다고 맞는 것이 아니다. 한두 샘플로 낸 값은 사실상 노이즈고, 0 이면 그
        지표는 None 이 된다. head_lin_acc 는 10 샘플 미만이면 버린다(current 참조).
        deque 길이가 아니라 창 안의 개수를 세는 이유는, _trim 이 push 때만 돌아서
        스트림이 끊기면 지난 샘플이 그대로 남아 있기 때문이다.
        """
        lo = now - self.W * C.QPC
        with self._lock:
            return {
                'visual_clutter': sum(t >= lo for t, _ in self._vis),
                'visual_flow': sum(t >= lo for t, _ in self._flow),
                'audio_density': sum(t >= lo for t, _ in self._aud),
                'gaze': sum(t >= lo for t, *_ in self._gaze),
                'head_lin_acc': sum(len(x) for t, x in self._acc if t >= lo),
                'head_ang_vel': sum(len(x) for t, x in self._gyr if t >= lo),
                'hand_L': sum(t >= lo for t, *_ in self._hand['L']),
                'hand_R': sum(t >= lo for t, *_ in self._hand['R']),
            }

    def current(self, now):
        with self._lock:  # 스냅샷만 lock 안에서
            acc = [(t, x) for (t, x) in self._acc]
            gyr = [(t, x) for (t, x) in self._gyr]
            hand = {s: list(self._hand[s]) for s in ('L', 'R')}
            vis = list(self._vis)
            flow = list(self._flow)
            aud = list(self._aud)
            gaze = list(self._gaze)

        lo = now - self.W * C.QPC
        hl = C.HALF_LIFE
        out = {}

        def wmean(pairs):  # [(ts, val)] -> recency 가중 평균
            seg = [(t, c) for (t, c) in pairs if t >= lo]
            if not seg:
                return np.nan
            return C._weighted_stat([t for t, _ in seg], [c for _, c in seg], now, hl, 'mean')

        out['visual_clutter'] = wmean(vis)
        out['visual_flow'] = wmean(flow)
        out['audio_density'] = wmean(aud)

        gz = np.array([(y, p) for (t, y, p) in gaze if t >= lo], float)
        if len(gz) >= 3:
            c = np.median(gz, axis=0)
            dev = np.sqrt(np.sum((gz - c) ** 2, axis=1))
            out['gaze_concentration'] = float(np.mean(dev <= 5.0))
            out['gaze_spread'] = float(np.sqrt(np.mean(dev ** 2)))
        else:
            out['gaze_concentration'] = np.nan
            out['gaze_spread'] = np.nan

        # head 선형가속도/각속도: 패킷 ts를 샘플별로 펼쳐 recency 가중 RMS
        acc_w = [(t, x) for (t, x) in acc if t >= lo]
        if acc_w and sum(len(x) for _, x in acc_w) >= 10:
            A = np.vstack([x for _, x in acc_w])
            sts = np.concatenate([np.full(len(x), t) for t, x in acc_w])
            lin = A - C.moving_average(A, max(1, A.shape[0] // 2)) if self.accel_has_gravity else A
            out['head_lin_acc'] = C._weighted_stat(sts, np.linalg.norm(lin, axis=1), now, hl, 'rms')
        else:
            out['head_lin_acc'] = np.nan

        gyr_w = [(t, x) for (t, x) in gyr if t >= lo]
        if gyr_w:
            G = np.vstack([x for _, x in gyr_w])
            sts = np.concatenate([np.full(len(x), t) for t, x in gyr_w])
            out['head_ang_vel'] = C._weighted_stat(sts, np.linalg.norm(G, axis=1), now, hl, 'rms')
        else:
            out['head_ang_vel'] = np.nan

        for side in ('L', 'R'):
            H = [r for r in hand[side] if r[0] >= lo]
            out[f'hand_acc_{side}'] = self._hand_acc(H, now)
        return out

    @staticmethod
    def _hand_acc(H, now, recency=0.3):
        """손 없으면 0 (최근 recency초 유효 손 없으면 미존재). 존재할 때만 가속도 RMS."""
        if len(H) < 12:
            return 0.0
        ts_arr = np.array([r[0] for r in H], np.int64)
        pos = np.array([(r[1], r[2], r[3]) for r in H], float)
        valid = np.array([r[4] for r in H], bool)
        vts = ts_arr[valid]
        if vts.size == 0 or (now - vts[-1]) > recency * C.QPC:
            return 0.0  # 현재 미존재
        acc = haptic._hand_accel_mag(pos, ts_arr / C.QPC, valid)
        m = ~np.isnan(acc)
        if not m.any():
            return 0.0
        return C._weighted_stat(ts_arr[m], acc[m], now, C.HALF_LIFE, 'rms')
