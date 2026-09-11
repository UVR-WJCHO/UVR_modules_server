"""
head 위치 + 방향 시각화 (라이브 녹화·재생 공용).

두 직교 뷰를 나란히 그려 3D 방향을 파악하기 쉽게 한다:
  - TOP  (X-Z): 위에서 본 뷰. 수평 이동 + yaw(좌우 방향)
  - RIGHT(Z-Y): 오른쪽에서 본 뷰. 높이 + pitch(상하 방향)
각 뷰에 head 위치(점)·이동 궤적·forward 화살표를 그린다. world 좌표(app 시작 원점, Y-up).
"""

import numpy as np
import cv2


def _view(pos, forward, trail, size, span, hx, hsign, vx, vsign, title, hud):
    """한 직교 뷰 패널. hx/vx = 화면 가로/세로에 쓸 world 축 index, hsign/vsign = 부호."""
    img = np.full((size, size, 3), 30, np.uint8)
    c = size // 2
    scale = size / span

    def px(p):
        return int(c + hsign * p[hx] * scale), int(c + vsign * p[vx] * scale)

    step = max(1, int(scale * 0.5))  # 0.5m 격자
    for g in range(0, size, step):
        cv2.line(img, (g, 0), (g, size), (50, 50, 50), 1)
        cv2.line(img, (0, g), (size, g), (50, 50, 50), 1)
    cv2.drawMarker(img, (c, c), (90, 90, 90), cv2.MARKER_CROSS, 16, 1)
    cv2.putText(img, title, (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    if trail is not None and len(trail) >= 2:
        pts = [px(p) for p in trail]
        for k in range(1, len(pts)):
            cv2.line(img, pts[k - 1], pts[k], (120, 120, 60), 1)

    if pos is None or forward is None:
        cv2.putText(img, 'head: invalid', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return img

    hp = px(pos)
    d = np.array([hsign * forward[hx], vsign * forward[vx]], float)
    n = np.linalg.norm(d)
    if n > 1e-6:
        d /= n
        tip = (int(hp[0] + d[0] * scale * 0.6), int(hp[1] + d[1] * scale * 0.6))
        cv2.arrowedLine(img, hp, tip, (0, 200, 255), 2, tipLength=0.3)
    cv2.circle(img, hp, 6, (0, 0, 255), -1)

    for i, line in enumerate(hud):
        cv2.putText(img, line, (8, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


def render_head_view(pos, forward, trail=None, size=360, span=3.0):
    """TOP + RIGHT 두 뷰를 가로로 합친 BGR 이미지 반환.

    pos, forward: world 좌표 (3,) 또는 None(무효). trail: 최근 위치들의 시퀀스.
    """
    if pos is not None and forward is not None:
        p = np.asarray(pos, float)
        f = np.asarray(forward, float)
        yaw = np.degrees(np.arctan2(f[0], -f[2]))
        pitch = np.degrees(np.arcsin(np.clip(f[1] / (np.linalg.norm(f) + 1e-9), -1, 1)))
        top_hud = [f'X{p[0]:+.2f} Y{p[1]:+.2f} Z{p[2]:+.2f}', f'yaw {yaw:+.0f}']
        side_hud = [f'pitch {pitch:+.0f}']
    else:
        top_hud = side_hud = []

    # TOP: 화면 오른쪽=+X, 위=-Z(정면).  RIGHT: 화면 오른쪽=-Z(정면), 위=+Y.
    top = _view(pos, forward, trail, size, span, 0, +1, 2, +1, 'TOP  X-Z (up=-Z fwd)', top_hud)
    side = _view(pos, forward, trail, size, span, 2, -1, 1, -1, 'RIGHT  Z-Y (up=+Y, fwd->)', side_hud)
    div = np.full((size, 3, 3), 80, np.uint8)
    return np.hstack([top, div, side])
