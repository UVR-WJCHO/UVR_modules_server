"""Visual load: 시야 안 시각적 복잡도.

- visual_edge_density : Canny 엣지 픽셀 비율 (구조적 밀집도)
- visual_entropy      : 그래디언트 크기 분포 엔트로피 (텍스처 복잡도, 0–1)
- visual_clutter      : 위 둘의 평균 (정적 복잡도 종합)

동적 복잡도(optical flow)는 visual_flow로 별도 추가 예정.
RGB 프레임(파일명=QPC)마다 계산 후 출력 타임라인에 윈도우 평균.
"""

import os
import glob
import numpy as np
import cv2

from . import common as C


def _frame_ts(path):
    return int(os.path.basename(path).split('_')[1].split('.')[0])


TEX_REF = 300.0  # 텍스처 에너지 정규화 기준(평균 gradient magnitude)


def clutter_value(gray, size=256):
    """그레이 이미지의 (edge_density, texture_energy). 라이브/오프라인 공용.

    - edge_density  : Canny 엣지 픽셀 비율 (0–1). "시야의 몇 %가 윤곽인가". 노이즈 강건.
    - texture_energy: clip(평균|grad| / TEX_REF, 0, 1). 엣지가 못 잡는 미세 텍스처.
    둘 다 장면이 복잡할수록 단조 증가(빈 벽+노이즈 ≈ 0).
    """
    h, w = gray.shape
    s = size / max(h, w)
    if s < 1.0:
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    img = gray

    edge_density = float(cv2.Canny(img, 50, 150).mean() / 255.0)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    texture = float(min(1.0, np.sqrt(gx * gx + gy * gy).mean() / TEX_REF))

    return edge_density, texture


FLOW_SIZE = (160, 90)  # optical flow 계산용 다운스케일


def to_small(gray, size=FLOW_SIZE):
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def flow_residual(prev_small, cur_small):
    """Farneback dense flow에서 ego-motion(전역 median)을 뺀 잔차 평균 + ego 크기 [px/frame].

    잔차 = 장면 자체의 움직임(독립 이동/변화), ego = 머리/전역 움직임. 라이브·오프라인 공용.
    """
    fl = cv2.calcOpticalFlowFarneback(prev_small, cur_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    ego = np.median(fl.reshape(-1, 2), axis=0)
    res = np.linalg.norm(fl - ego, axis=2)
    return float(res.mean()), float(np.linalg.norm(ego))


def compute(session, out_ts, W):
    frames = sorted(glob.glob(os.path.join(session, 'rgb', 'frame_*.png')), key=_frame_ts)
    if not frames:
        return {}

    ts = np.array([_frame_ts(f) for f in frames], dtype=np.int64)
    n = len(frames)
    edge = np.full(n, np.nan); tex = np.full(n, np.nan)
    flow = np.full(n, np.nan); ego = np.full(n, np.nan)  # px/s (ego-motion 보정 잔차 / 전역)
    prev, prev_ts = None, None
    for i, f in enumerate(frames):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        edge[i], tex[i] = clutter_value(g)
        small = to_small(g)
        if prev is not None:
            dt = (ts[i] - prev_ts) / C.QPC
            if 0 < dt < 0.2:  # 인접 프레임만
                r, e = flow_residual(prev, small)
                flow[i] = r / dt; ego[i] = e / dt
        prev, prev_ts = small, ts[i]

    return {
        'visual_clutter': C.windowed(ts, edge, out_ts, W, 'mean'),   # = edge density
        'visual_texture': C.windowed(ts, tex, out_ts, W, 'mean'),
        'visual_flow': C.windowed(ts, flow, out_ts, W, 'mean'),
        'visual_egomotion': C.windowed(ts, ego, out_ts, W, 'mean'),
    }
