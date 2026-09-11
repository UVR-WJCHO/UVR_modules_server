"""정량 지표 패널 렌더러 (라이브 녹화·재생 공용).

지표를 Visual / Audio / Haptic 카테고리로 묶어 가로 막대로 그린다. 라이브·재생 모두
동일하게 쓰도록, 세션 전체 통계가 아니라 지표별 고정 표시범위(display max)로 막대를 채운다.
표시하는 값은 라이브 스트리밍으로 계산 가능한 것들(StreamingMetrics.current)로 한정.
"""

import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

# (카테고리, [(key, 라벨, 표시상한), ...])  — 막대 = clip(value / max, 0, 1)
PANEL_GROUPS = [
    ('VISUAL', [
        ('visual_clutter',     'Clutter',  0.3),
        ('visual_flow',        'Motion',   50.0),
        ('gaze_concentration', 'Gaze foc', 1.0),
    ]),
    ('AUDIO', [
        ('audio_density',      'Density',  5.0),
    ]),
    ('HAPTIC', [
        ('head_lin_acc',       'Head acc', 2.0),
        ('head_ang_vel',       'Head rot', 3.0),
        ('hand_acc_R',         'Hand R',   20.0),
        ('hand_acc_L',         'Hand L',   20.0),
    ]),
]

CAT_COLOR = {'VISUAL': (200, 150, 200), 'AUDIO': (150, 150, 230), 'HAPTIC': (230, 190, 150)}


def _bar_color(frac):
    if frac < 0.5:
        return (0, 200, 0)      # green
    if frac < 0.8:
        return (0, 200, 220)    # yellow
    return (0, 0, 220)          # red


def render_metrics_panel(values, width=300, row_h=34, head_h=24, pad=12, title='LOAD METRICS'):
    """values: {key: raw value}. 반환: BGR 패널 이미지."""
    n_rows = sum(len(m) for _, m in PANEL_GROUPS)
    height = pad * 2 + 24 + len(PANEL_GROUPS) * head_h + n_rows * row_h
    img = np.full((height, width, 3), 35, np.uint8)
    cv2.putText(img, title, (pad, pad + 12), FONT, 0.55, (220, 220, 220), 1)

    bar_x = 96
    bar_w = width - bar_x - pad
    y = pad + 26
    for cat, metrics in PANEL_GROUPS:
        cv2.putText(img, cat, (pad, y + 15), FONT, 0.48, CAT_COLOR.get(cat, (180, 180, 180)), 1)
        cv2.line(img, (pad, y + 20), (width - pad, y + 20), (70, 70, 70), 1)
        y += head_h
        for key, label, vmax in metrics:
            v = values.get(key, np.nan)
            cv2.putText(img, label, (pad + 6, y + 16), FONT, 0.42, (200, 200, 200), 1)
            cv2.rectangle(img, (bar_x, y + 2), (bar_x + bar_w, y + 19), (60, 60, 60), -1)
            if v == v:  # not NaN
                frac = float(np.clip(v / vmax, 0.0, 1.0))
                cv2.rectangle(img, (bar_x, y + 2), (int(bar_x + bar_w * frac), y + 19), _bar_color(frac), -1)
                cv2.putText(img, f'{v:.2f}', (bar_x + bar_w - 46, y + 16), FONT, 0.4, (255, 255, 255), 1)
            else:
                cv2.putText(img, '--', (bar_x + 4, y + 16), FONT, 0.4, (120, 120, 120), 1)
            y += row_h
    return img
