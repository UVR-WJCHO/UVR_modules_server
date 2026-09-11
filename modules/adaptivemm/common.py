"""지표 추출 공통 유틸: 로더 / 타임라인·윈도우 / 신호 / egocentric / 정규화 / IO.

시간 단위는 HoloLens QPC(100 ns) 정수 타임스탬프. 초 = QPC / 1e7.
설계: docs/metrics_implementation_note.md
"""

import os
import json
import numpy as np

QPC = 1e7            # 100 ns ticks per second
WRIST = 1            # SI_HandJointKind.Wrist
PALM = 0             # SI_HandJointKind.Palm


# ---------------------------------------------------------------- 로더
def load_si(session):
    return np.load(os.path.join(session, 'spatial_input', 'si_data.npz'))


def load_imu(session, name):
    """IMU csv -> (ts_qpc[N] int64, xyz[N,3] float).

    per-sample 절대시각은 packet timestamp(QPC)를 사용(패킷 간격 ~수십 ms << 윈도우 2 s라 충분).
    columns: packet_id,timestamp,sample_idx,soc_ticks,vinyl_hup_ticks,x,y,z,temperature
    """
    path = os.path.join(session, 'imu', f'{name}.csv')
    d = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(1, 5, 6, 7))
    return d[:, 0].astype(np.int64), d[:, 1:4]


def session_span(session):
    ts = load_si(session)['timestamp']
    return int(ts[0]), int(ts[-1])


def output_timeline(session, rate):
    """세션 구간을 rate(Hz)로 균일 샘플한 출력 타임라인(QPC int64)."""
    t0, t1 = session_span(session)
    n = int((t1 - t0) / QPC * rate) + 1
    return (t0 + (np.arange(n) / rate * QPC)).astype(np.int64)


# ---------------------------------------------------------------- 신호 유틸
def moving_average(x, w):
    """축0 방향 박스 필터(edge 패딩, 길이 보존). x: (N,) 또는 (N,C)."""
    if w <= 1:
        return x
    k = np.ones(w) / w
    pad = (w // 2, w - 1 - w // 2)
    if x.ndim == 1:
        return np.convolve(np.pad(x, pad, 'edge'), k, 'valid')
    return np.stack([np.convolve(np.pad(x[:, c], pad, 'edge'), k, 'valid')
                     for c in range(x.shape[1])], axis=1)


def gravity_removed(accel_xyz, fs, cutoff_s=1.0):
    """accel(m/s², 중력포함)에서 저역(≈중력)을 빼 동적 가속도만 남김."""
    w = max(1, int(fs * cutoff_s))
    return accel_xyz - moving_average(accel_xyz, w)


# ---------------------------------------------------------------- 윈도우 통계
def _agg(x, stat):
    if stat == 'rms':
        return float(np.sqrt(np.mean(x * x)))
    if stat == 'mean':
        return float(x.mean())
    if stat == 'std':
        return float(x.std())
    if stat == 'p90':
        return float(np.percentile(x, 90))
    if stat == 'max':
        return float(x.max())
    return float(x.mean())


HALF_LIFE = 0.15  # recency 가중 반감기(초). 작을수록 최근값에 민감(지연↓). mean/rms에만 적용.


def _weighted_stat(sample_ts, vals, now, half_life, stat):
    """recency 가중 통계(반감기 half_life초). 최근 표본일수록 큰 가중 → 지연 감소."""
    vals = np.asarray(vals, float)
    if stat not in ('mean', 'rms') or not half_life:
        return _agg(vals, stat)
    w = 0.5 ** ((now - np.asarray(sample_ts, float)) / (half_life * QPC))
    sw = w.sum()
    if sw <= 1e-12:
        return _agg(vals, stat)
    if stat == 'rms':
        return float(np.sqrt(np.sum(w * vals * vals) / sw))
    return float(np.sum(w * vals) / sw)


def windowed_gated(src_ts, src_val, present_ts, out_ts, W, recency, stat='rms', half_life=HALF_LIFE):
    """windowed와 같되, [now-recency, now] 구간에 present 표본이 없으면 0(현재 미존재).

    present_ts: 대상이 '존재'한(예: 손 추적 유효) 시각들(정렬). 손이 사라지면 즉시 0으로
    떨어지고, 존재할 때만 recency 가중 값을 낸다.
    """
    out = np.zeros(len(out_ts))
    Wt = int(W * QPC); Rt = int(recency * QPC)
    for i, t in enumerate(out_ts):
        if np.searchsorted(present_ts, t, 'right') <= np.searchsorted(present_ts, t - Rt):
            continue  # 최근 미존재 -> 0
        a = np.searchsorted(src_ts, t - Wt)
        b = np.searchsorted(src_ts, t, 'right')
        seg_ts = src_ts[a:b]; seg = src_val[a:b]
        m = ~np.isnan(seg)
        if m.any():
            out[i] = _weighted_stat(seg_ts[m], seg[m], t, half_life, stat)
    return out


def windowed(src_ts, src_val, out_ts, W, stat='rms', mode='causal', half_life=HALF_LIFE):
    """각 출력시각 t마다 [t-W, t](causal) 구간의 recency 가중 통계.

    src_ts: 정렬된 QPC int64, src_val: (N,) (NaN 허용, 제외됨). 반환: (len(out_ts),).
    """
    out = np.full(len(out_ts), np.nan)
    Wt = int(W * QPC)
    hl = half_life if mode == 'causal' else None  # centered면 가중 없음
    for i, t in enumerate(out_ts):
        lo, hi = (t - Wt, t) if mode == 'causal' else (t - Wt // 2, t + Wt // 2)
        a = np.searchsorted(src_ts, lo)
        b = np.searchsorted(src_ts, hi, 'right')
        seg_ts = src_ts[a:b]; seg = src_val[a:b]
        m = ~np.isnan(seg)
        if m.any():
            out[i] = _weighted_stat(seg_ts[m], seg[m], t, hl, stat)
    return out


# ---------------------------------------------------------------- egocentric
def head_frame(forward, up):
    """head 좌표축(right,up,forward) 행벡터로 구성한 R (world->head, 3x3).

    HL2 world는 forward가 -Z라, right = cross(forward, up)로 해야 해부학적 오른쪽(+x)이 된다.
    """
    f = forward / (np.linalg.norm(forward) + 1e-9)
    u = up / (np.linalg.norm(up) + 1e-9)
    r = np.cross(f, u); r /= (np.linalg.norm(r) + 1e-9)
    u2 = np.cross(r, f)
    return np.stack([r, u2, f], axis=0)


def to_egocentric(points_world, head_pos, forward, up):
    """world 점들을 head 기준 좌표로. points_world:(...,3)."""
    return (points_world - head_pos) @ head_frame(forward, up).T


def gaze_yaw_pitch(eye_dir, forward, up):
    """시선 방향(world)을 head 기준으로 변환한 yaw/pitch [deg]. 정면=(0,0).

    yaw: 우+/좌−, pitch: 상+/하−. eye_dir/forward/up: (3,).
    """
    g = head_frame(forward, up) @ np.asarray(eye_dir, float)  # [·right, ·up, ·forward]
    yaw = np.degrees(np.arctan2(g[0], g[2]))
    pitch = np.degrees(np.arcsin(np.clip(g[1], -1.0, 1.0)))
    return yaw, pitch


# ---------------------------------------------------------------- 정규화
def robust_z(x):
    """robust z-score: (x - median) / (IQR/1.349). 반환 (z, stats)."""
    x = np.asarray(x, float)
    if np.all(np.isnan(x)):
        return np.zeros_like(x), {'median': 0.0, 'scale': 1.0}
    m = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    s = iqr / 1.349 if iqr > 1e-9 else (np.nanstd(x) or 1.0)
    return (x - m) / s, {'median': float(m), 'scale': float(s)}


# ---------------------------------------------------------------- IO
def write_metrics(session, out_ts, columns, meta, rate, window):
    """metrics.csv + metrics_meta.json 저장. columns: name->array(len=out_ts)."""
    names = list(columns.keys())
    tsec = (out_ts - out_ts[0]) / QPC
    arr = np.column_stack([out_ts.astype(np.float64), tsec] + [columns[n] for n in names])
    header = ','.join(['timestamp', 't_sec'] + names)
    path = os.path.join(session, 'metrics.csv')
    np.savetxt(path, arr, delimiter=',', header=header, comments='', fmt='%.6f')

    meta.update({'rate_hz': rate, 'window_s': window,
                 'columns': names, 'n_rows': int(len(out_ts))})
    with open(os.path.join(session, 'metrics_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path
