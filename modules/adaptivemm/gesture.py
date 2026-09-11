"""Gesture 활동 영역: 일정 시간 동안 주된 손의 공간적 위치.

손목 위치를 head-egocentric 좌표(오른쪽 x, 위 y, 앞 z)로 변환해, 손별로 윈도우 W 동안:
- hand_cx/cy/cz_{L,R} : 주 활동 영역 중심 (ego m, median)
- hand_spread_{L,R}   : 위치 분산 RMS (m)
- hand_active_{L,R}   : 유효(추적) 비율 (0–1)
"""

import numpy as np
from . import common as C


def _ego_batch(pos_world, head_pos, forward, up):
    """프레임별 손목 ego 좌표 (N,3). head_frame이 프레임마다 달라 루프."""
    n = len(pos_world)
    ego = np.full((n, 3), np.nan)
    for i in range(n):
        ego[i] = C.to_egocentric(pos_world[i], head_pos[i], forward[i], up[i])
    return ego


def compute(session, out_ts, W):
    si = C.load_si(session)
    ts = si['timestamp'].astype(np.int64)
    hp, fwd, up = si['head_position'], si['head_forward'], si['head_up']
    head_v = si['head_valid']

    cols = {}
    Wt = int(W * C.QPC)
    for side, pkey, vkey in [('L', 'hand_left_position', 'hand_left_valid'),
                             ('R', 'hand_right_position', 'hand_right_valid')]:
        valid = si[vkey] & head_v
        ego = _ego_batch(si[pkey][:, C.WRIST, :].astype(float), hp, fwd, up)
        ego[~valid] = np.nan

        K = len(out_ts)
        cx = np.full(K, np.nan); cy = np.full(K, np.nan); cz = np.full(K, np.nan)
        spread = np.full(K, np.nan); active = np.zeros(K)
        for k, t in enumerate(out_ts):
            a = np.searchsorted(ts, t - Wt)
            b = np.searchsorted(ts, t, 'right')
            seg = ego[a:b]
            n_win = b - a
            m = ~np.isnan(seg[:, 0])
            active[k] = (m.sum() / n_win) if n_win > 0 else 0.0
            pts = seg[m]
            if pts.shape[0] >= 3:
                c = np.median(pts, axis=0)
                cx[k], cy[k], cz[k] = c
                spread[k] = float(np.sqrt(np.mean(np.sum((pts - c) ** 2, axis=1))))

        cols[f'hand_cx_{side}'] = cx
        cols[f'hand_cy_{side}'] = cy
        cols[f'hand_cz_{side}'] = cz
        cols[f'hand_spread_{side}'] = spread
        cols[f'hand_active_{side}'] = active
    return cols
