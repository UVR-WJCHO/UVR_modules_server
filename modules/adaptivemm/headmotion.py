"""head pose 시계열에서 선형가속도와 각속도를 만든다.

기기가 IMU 를 보내지 않으므로(HL2 accel/gyro 가 그 앱의 C# 경로로 들어오지 않는다),
매 패킷에 이미 들어 있는 `head_pos_*` 와 `head_rot_*` 를 미분해서 쓴다.

    가속도  위치 2차 미분                      m/s^2
    각속도  쿼터니언 차분의 벡터부 * 2 / dt     rad/s

여기서 나오는 가속도에는 **중력이 없다.** 가속도계가 아니라 기구학적 미분이기 때문이다.
정지한 머리의 값은 0 이지 1 g 가 아니다. 그래서 이 소스를 쓸 때는
`StreamingMetrics(accel_has_gravity=False)` 로 두어야 한다. 그러지 않으면 moving average
가 실제 저주파 운동을 빼버린다.

한계 두 가지를 기록해 둔다.

  - 전송이 30 Hz 라 나이퀴스트 15 Hz 까지만 본다. 진동·떨림 같은 고주파는 원리적으로
    볼 수 없다. 진짜 IMU 는 수백 Hz 다.
  - 2 차 미분이라 노이즈가 증폭된다. 그래서 미분 전에 짧은 이동평균으로 평활한다.
    각속도는 1 차라 훨씬 안정적이다.
"""
from __future__ import annotations

from collections import deque

import numpy as np

SMOOTH_N = 3            # 미분 전 이동평균 길이(프레임). 2차 미분의 노이즈를 누른다
MIN_DT = 1e-4           # 초. 이보다 짧은 간격은 같은 프레임으로 보고 버린다
MAX_DT = 0.5            # 초. 이보다 벌어지면 연속이 아니라고 보고 체인을 끊는다


def _quat_mul(a, b):
    """(x, y, z, w) 규약. a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], np.float64)


def _quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], np.float64)


class HeadMotion:
    """패킷마다 pose 를 넣으면 (선형가속도, 각속도)를 돌려준다.

    준비가 안 된 동안(샘플 부족, 간격 과다)에는 None 을 돌려준다. 호출자는 None 이면
    그 프레임을 건너뛰면 된다.
    """

    def __init__(self, smooth_n: int = SMOOTH_N):
        self._pos = deque(maxlen=smooth_n)      # 평활용 원본 (t, 위치)
        self._hist = deque(maxlen=3)            # (t_sec, smoothed_pos, quat)

    def reset(self) -> None:
        self._pos.clear()
        self._hist.clear()

    def push(self, t_sec: float, position, rotation):
        """position (3,) m, rotation (4,) 쿼터니언 (x, y, z, w).

        반환 (lin_acc (3,), ang_vel (3,)) 또는 None.
        """
        p = np.asarray(position, np.float64)
        q = np.asarray(rotation, np.float64)
        n = np.linalg.norm(q)
        if not np.isfinite(p).all() or not np.isfinite(q).all() or n < 1e-6:
            return None
        q = q / n

        if self._pos and not (MIN_DT < t_sec - self._pos[-1][0] < MAX_DT):
            # 간격이 없거나 과하게 벌어졌다. 미분을 이어 붙이면 큰 가짜 값이 나온다.
            self.reset()

        # 위치를 평균 낼 때 시각도 같이 평균 낸다. 위치만 평균 내고 현재 시각과 짝지으면
        # 간격이 불규칙할 때 그 어긋남이 2차 미분에 편향으로 들어온다(등가속도 1.0 이
        # 1.9 로 나오는 것을 확인했다). 간격이 일정하면 어느 쪽이든 같다.
        self._pos.append((t_sec, p))
        t_bar = float(np.mean([s for s, _ in self._pos]))
        p_bar = np.mean([v for _, v in self._pos], axis=0)
        self._hist.append((t_bar, p_bar, q))
        if len(self._hist) < 3:
            return None

        (t0, p0, _), (t1, p1, q1), (t2, p2, q2) = self._hist

        # 각속도: 인접 두 회전의 차분. dq 의 부호를 w>0 쪽으로 맞춰 짧은 쪽 회전을 쓴다.
        dq = _quat_mul(q2, _quat_conj(q1))
        if dq[3] < 0:
            dq = -dq
        ang_vel = 2.0 * dq[:3] / (t2 - t1)

        # 가속도: 불균일 간격을 반영한 2차 차분
        h1, h2 = t1 - t0, t2 - t1
        lin_acc = 2.0 * (h1 * p2 - (h1 + h2) * p1 + h2 * p0) / (h1 * h2 * (h1 + h2))

        return lin_acc.astype(np.float32), ang_vel.astype(np.float32)
