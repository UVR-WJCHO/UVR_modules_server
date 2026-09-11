"""Gaze 활동 영역: 일정 시간 동안 주된 시선 집중 영역.

시선(SI eye_ray)을 head 기준(eye-in-head) yaw/pitch로 변환해, 윈도우 W 동안:
- gaze_center_yaw / gaze_center_pitch : 주 집중 영역 중심 (deg, median)
- gaze_spread        : 각 분산 RMS (deg)
- gaze_concentration : 중심 CONE_DEG 이내 비율 (0–1, 높을수록 고정/집중)
"""

import numpy as np
from . import common as C

CONE_DEG = 5.0  # 집중도 판정 반경


def gaze_in_head_batch(eye_dir, forward, up):
    """프레임별 gaze yaw/pitch [deg] (head 기준). 각 (N,3) -> (N,), (N,)."""
    def unit(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    f = unit(forward.astype(float))
    u = unit(up.astype(float))
    r = np.cross(f, u); r /= (np.linalg.norm(r, axis=1, keepdims=True) + 1e-9)
    u2 = np.cross(r, f)
    d = eye_dir.astype(float)
    gx = np.sum(d * r, axis=1)
    gy = np.sum(d * u2, axis=1)
    gz = np.sum(d * f, axis=1)
    yaw = np.degrees(np.arctan2(gx, gz))
    pitch = np.degrees(np.arcsin(np.clip(gy, -1.0, 1.0)))
    return yaw, pitch


def compute(session, out_ts, W):
    si = C.load_si(session)
    ts = si['timestamp'].astype(np.int64)
    valid = si['eye_valid'] & si['head_valid']
    yaw, pitch = gaze_in_head_batch(si['eye_direction'], si['head_forward'], si['head_up'])
    yaw[~valid] = np.nan
    pitch[~valid] = np.nan

    K = len(out_ts)
    cy = np.full(K, np.nan); cp = np.full(K, np.nan)
    spread = np.full(K, np.nan); conc = np.full(K, np.nan)
    Wt = int(W * C.QPC)
    for k, t in enumerate(out_ts):
        a = np.searchsorted(ts, t - Wt)
        b = np.searchsorted(ts, t, 'right')
        yy = yaw[a:b]; pp = pitch[a:b]
        m = ~np.isnan(yy)
        yy = yy[m]; pp = pp[m]
        if yy.size >= 3:
            my = float(np.median(yy)); mp = float(np.median(pp))
            dev = np.sqrt((yy - my) ** 2 + (pp - mp) ** 2)
            cy[k] = my; cp[k] = mp
            spread[k] = float(np.sqrt(np.mean(dev ** 2)))
            conc[k] = float(np.mean(dev <= CONE_DEG))
    return {
        'gaze_center_yaw': cy, 'gaze_center_pitch': cp,
        'gaze_spread': spread, 'gaze_concentration': conc,
    }
