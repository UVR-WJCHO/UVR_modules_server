"""Haptic load: head/hand 물리적 움직임 부하.

- head_lin_acc : IMU 가속도(중력 제거) 크기의 윈도우 RMS  [m/s²]
- head_ang_vel : IMU 자이로 각속도 크기의 윈도우 RMS       [rad/s]
- hand_acc_L/R : 손목 위치 2차미분 가속도 크기의 윈도우 RMS [m/s²]
"""

import os
import numpy as np
from . import common as C


def _hand_accel_mag(pos, ts_sec, valid, smooth=7, max_gap=0.05, max_speed=4.0, min_run=8, edge=2):
    """손목 위치(N,3) 가속도 크기 [m/s²]. 트래킹 글리치/등장·이탈 스파이크 억제.

    연속 유효 구간(인접 프레임, dt<max_gap)만 처리하고, 그 길이가 min_run 이상일 때만
    구간 내 위치를 스무딩 후 2차 중심차분. 추가로:
    - 구간 양끝 edge프레임 제외 → 손 등장/이탈 순간(FOV 가장자리 불안정) 가속도 배제.
    - 순간속도가 max_speed(사람 손 한계, m/s)를 넘는 프레임은 트래킹 점프로 보고 제외.
    """
    n = len(pos)
    acc = np.full(n, np.nan)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i  # 연속 유효 구간 [i, j]
        while (j + 1 < n) and valid[j + 1] and (ts_sec[j + 1] - ts_sec[j] < max_gap):
            j += 1
        L = j + 1 - i
        if L >= min_run:
            ps = C.moving_average(pos[i:j + 1], min(smooth, L))
            tt = ts_sec[i:j + 1]
            for k in range(1 + edge, L - 1 - edge):  # 양끝 edge 제외
                span = tt[k + 1] - tt[k - 1]
                if span <= 0:
                    continue
                if np.linalg.norm(ps[k + 1] - ps[k - 1]) / span > max_speed:
                    continue  # 트래킹 점프
                a = (ps[k + 1] - 2 * ps[k] + ps[k - 1]) / ((span / 2) ** 2)
                acc[i + k] = float(np.linalg.norm(a))
        i = j + 1
    return acc


def compute(session, out_ts, W):
    cols = {}

    # head 선형 가속도 (중력 제거)
    if os.path.isfile(os.path.join(session, 'imu', 'accelerometer.csv')):
        a_ts, a_xyz = C.load_imu(session, 'accelerometer')
        fs = len(a_ts) / max(1e-6, (a_ts[-1] - a_ts[0]) / C.QPC)
        lin_mag = np.linalg.norm(C.gravity_removed(a_xyz, fs, cutoff_s=1.0), axis=1)
        cols['head_lin_acc'] = C.windowed(a_ts, lin_mag, out_ts, W, 'rms')

    # head 각속도 (자이로)
    if os.path.isfile(os.path.join(session, 'imu', 'gyroscope.csv')):
        g_ts, g_xyz = C.load_imu(session, 'gyroscope')
        cols['head_ang_vel'] = C.windowed(g_ts, np.linalg.norm(g_xyz, axis=1), out_ts, W, 'rms')

    # hand 가속도 (손목 위치 미분)
    si = C.load_si(session)
    ts = si['timestamp'].astype(np.int64)
    ts_sec = ts / C.QPC
    for side, pkey, vkey in [('L', 'hand_left_position', 'hand_left_valid'),
                             ('R', 'hand_right_position', 'hand_right_valid')]:
        valid = si[vkey]
        pos = si[pkey][:, C.WRIST, :].astype(float)
        acc_mag = _hand_accel_mag(pos, ts_sec, valid)
        # 손이 최근 0.3초 내 없으면 0 (존재할 때만 값)
        cols[f'hand_acc_{side}'] = C.windowed_gated(ts, acc_mag, ts[valid], out_ts, W, 0.3, 'rms')

    return cols
