import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """두 마스크 간의 IoU(Intersection over Union) 계산
    
    Args:
        mask1: bool array (H, W)
        mask2: bool array (H, W)
    
    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def bbox_xywh_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """COCO-style bbox [x, y, w, h] from mask. Returns None if mask is empty."""
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    w = int(x2 - x1 + 1)
    h = int(y2 - y1 + 1)
    return (int(x1), int(y1), w, h)


def mask_to_coco_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Convert binary mask to COCO-style uncompressed RLE (column-major)."""
    if mask is None:
        return {"counts": [], "size": [0, 0]}
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape[:2]
    # COCO expects column-major order (Fortran order)
    flat = mask_u8.T.reshape(-1)
    counts: List[int] = []
    prev = 0
    run = 0
    for v in flat:
        if int(v) == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = int(v)
    counts.append(run)
    return {"counts": counts, "size": [int(h), int(w)]}


def bbox_iou(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in box_a]
    bx1, by1, bx2, by2 = [float(x) for x in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1 + 1.0)
    ih = max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0))
    area_b = max(0.0, (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def dilate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return mask.astype(bool)
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def stable_angle(key: str) -> float:
    h = 2166136261
    for ch in key.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return (h % 360) * (np.pi / 180.0)


def calculate_ioa_bidirectional(
    mask1: np.ndarray,
    mask2: np.ndarray,
    min_area_for_ioa: int = 5,
) -> tuple[float, float, float]:
    """양방향 IoA (Intersection over Area) 계산
    
    작은 객체가 큰 객체에 포함되는 경우를 감지하기 위해 양방향으로 체크.
    표준 IoU는 대칭적이지만, IoA는 비대칭적이므로 양쪽 모두 확인 필요.
    
    예시:
        작은 마스크 A (100픽셀)가 큰 마스크 B (1000픽셀) 안에 90픽셀 포함:
        - IoU = 90/1010 = 0.089 (8.9%) ← 중복으로 보이지 않음!
        - IoA_A = 90/100 = 0.90 (90%) ← A는 거의 B에 포함됨!
        - IoA_B = 90/1000 = 0.09 (9%)
    
    Args:
        mask1: bool array (H, W)
        mask2: bool array (H, W)
    
    Returns:
        (iou, ioa_1, ioa_2) 튜플:
            - iou: 전통적 IoU (intersection / union)
            - ioa_1: mask1의 관점에서 IoA (intersection / area_mask1)
            - ioa_2: mask2의 관점에서 IoA (intersection / area_mask2)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    area1 = int(mask1.sum())
    area2 = int(mask2.sum())
    union = np.logical_or(mask1, mask2).sum()
    
    iou = float(intersection / union) if union > 0 else 0.0
    if area1 < min_area_for_ioa:
        ioa_1 = 0.0
    else:
        ioa_1 = float(intersection / area1) if area1 > 0 else 0.0
    if area2 < min_area_for_ioa:
        ioa_2 = 0.0
    else:
        ioa_2 = float(intersection / area2) if area2 > 0 else 0.0
    
    return iou, ioa_1, ioa_2


def is_duplicate_mask(mask1: np.ndarray, mask2: np.ndarray, 
                      iou_threshold: float = 0.5,
                      ioa_threshold: float = 0.8) -> bool:
    """두 마스크가 중복인지 양방향 체크
    
    IoA만 사용하여 중복 판단:
    - 포함 관계: IoA로 체크 (작은 객체가 큰 객체에 포함)
    
    Args:
        mask1: bool array (H, W)
        mask2: bool array (H, W)
        ioa_threshold: IoA 임계값 (기본 0.8, 80% 이상 포함되면 중복)
    
    Returns:
        중복 여부 (True/False)
    """
    _, ioa_1, ioa_2 = calculate_ioa_bidirectional(mask1, mask2)
    
    # 다음 중 하나라도 만족하면 중복으로 판단:
    # 1. mask1이 mask2에 대부분 포함됨
    # 2. mask2가 mask1에 대부분 포함됨
    return (ioa_1 >= ioa_threshold) and (ioa_2 >= ioa_threshold)


def visualize_segmentation(
    image_bgr: np.ndarray,
    masks: List[np.ndarray],
    boxes: List[List[int]],
    hands: Optional[List[Dict[str, Any]]] = None,
    labels: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.5,
    outline_only: bool = True,
    font_scale: float = 0.4,
) -> np.ndarray:
    """세그멘테이션 결과를 이미지에 시각화.

    Args:
        image_bgr: 원본 BGR 이미지
        masks: 세그멘테이션 마스크 리스트 (각각 H x W bool array)
        boxes: 바운딩 박스 리스트 [x1, y1, x2, y2]
        hands: 손 정보 리스트 (옵션, contact 정보 표시용)
        labels: 각 객체의 라벨 리스트 ("hand", "obj" 등)
        colors: 각 객체의 고정 색상 리스트 (옵션, None이면 기본 팔레트 사용)
        alpha: 오버레이 투명도
        outline_only: True이면 외곽선만, False이면 마스크 채우기
        font_scale: 폰트 크기 (기본 0.4)

    Returns:
        시각화된 BGR 이미지
    """
    output = image_bgr.copy()

    # 색상 팔레트 (colors가 제공되지 않은 경우 사용)
    default_colors = [
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 0, 255),    # red
        (255, 255, 0),  # cyan
        (255, 0, 255),  # magenta
        (0, 255, 255),  # yellow
    ]

    for idx, mask in enumerate(masks):
        # 고정 색상이 제공되면 사용, 아니면 기본 팔레트
        if colors and idx < len(colors):
            color = colors[idx]
        else:
            color = default_colors[idx % len(default_colors)]

        # 마스크 시각화
        if mask is not None and mask.any():
            if outline_only:
                # 외곽선만 그리기
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(output, contours, -1, color, 2)
            else:
                # 마스크 채우기 (기존 방식)
                overlay = output.copy()
                overlay[mask] = color
                output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

        # 바운딩 박스 (boxes가 제공된 경우에만)
        if boxes is not None and idx < len(boxes):
            box = boxes[idx]
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # 라벨
        if labels and idx < len(labels):
            label = labels[idx]
        elif hands and idx < len(hands):
            hand = hands[idx]
            lr = "L" if hand.get("lr", 0) == 0 else "R"
            contact = hand.get("contact_code", "?")
            label = f"{lr}-Hand -> Obj{idx} ({contact})"
        else:
            label = f"Obj {idx}"

        # 라벨 위치 결정 (bbox가 있으면 bbox 위, 없으면 마스크 영역 위)
        if boxes is not None and idx < len(boxes):
            box = boxes[idx]
            if box is not None:
                x1, y1, x2, y2 = box
                label_pos = (x1, y1 - 10)
            else:
                label_pos = (10, 30 + idx * 20)
        else:
            # 마스크의 상단 중심점 사용
            if mask is not None and mask.any():
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    label_pos = (int(xs.mean()), int(ys.min()) - 10)
                else:
                    label_pos = (10, 30 + idx * 20)
            else:
                label_pos = (10, 30 + idx * 20)
        
        cv2.putText(output, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    return output


def get_image_paths(image_dir: str, extensions: Optional[List[str]] = None) -> List[str]:
    """디렉토리에서 이미지 파일 경로를 가져옵니다.
    
    Args:
        image_dir: 이미지 디렉토리 경로
        extensions: 허용할 확장자 리스트 (기본: ['.jpg', '.jpeg', '.png', '.bmp'])
    
    Returns:
        정렬된 이미지 경로 리스트
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
    image_paths = []
    
    if not os.path.exists(image_dir):
        return image_paths
    
    for filename in os.listdir(image_dir):
        if any(filename.endswith(ext) for ext in extensions):
            image_paths.append(os.path.join(image_dir, filename))
    
    # 자연스러운 정렬 (숫자 고려)
    def natural_sort_key(path):
        basename = os.path.basename(path)
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split('([0-9]+)', basename)]
    
    image_paths.sort(key=natural_sort_key)
    
    return image_paths


def overlay_motion_debug(
    vis_image: np.ndarray,
    geom: Dict[int, Dict[str, Any]],
    prev_geom: Dict[int, Dict[str, Any]],
    a_bg: Optional[np.ndarray],
    bg_ok: bool,
) -> Dict[int, Dict[str, Any]]:
    bg_vec = None
    bg_mag = 0.0
    if isinstance(a_bg, np.ndarray):
        if a_bg.shape == (2, 3):
            bg_vec = (float(a_bg[0, 2]), float(a_bg[1, 2]))
        elif a_bg.shape == (3, 3):
            bg_vec = (float(a_bg[0, 2]), float(a_bg[1, 2]))
    if bg_vec is not None:
        bg_mag = float(np.hypot(bg_vec[0], bg_vec[1]))

    for oid, g in geom.items():
        c = g.get("centroid")
        if c is None:
            continue
        cx, cy = int(c[0]), int(c[1])
        if oid in prev_geom and prev_geom[oid].get("centroid") is not None:
            pc = prev_geom[oid]["centroid"]
            vx = int(cx - pc[0])
            vy = int(cy - pc[1])
            speed = float(np.hypot(vx, vy))
            cv2.arrowedLine(vis_image, (cx, cy), (cx + vx, cy + vy), (255, 255, 255), 1, tipLength=0.2)
            cv2.putText(
                vis_image, f"v={speed:.1f}", (cx + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA
            )
            if isinstance(a_bg, np.ndarray) and bg_vec is not None:
                if a_bg.shape == (2, 3):
                    ex = a_bg[0, 0] * pc[0] + a_bg[0, 1] * pc[1] + a_bg[0, 2]
                    ey = a_bg[1, 0] * pc[0] + a_bg[1, 1] * pc[1] + a_bg[1, 2]
                else:
                    w = a_bg[2, 0] * pc[0] + a_bg[2, 1] * pc[1] + a_bg[2, 2]
                    if abs(w) < 1e-6:
                        w = 1.0
                    ex = (a_bg[0, 0] * pc[0] + a_bg[0, 1] * pc[1] + a_bg[0, 2]) / w
                    ey = (a_bg[1, 0] * pc[0] + a_bg[1, 1] * pc[1] + a_bg[1, 2]) / w
                rvx = int(cx - ex)
                rvy = int(cy - ey)
                rs = float(np.hypot(rvx, rvy))
                cv2.arrowedLine(vis_image, (cx, cy), (cx + rvx, cy + rvy), (0, 200, 255), 1, tipLength=0.2)
                cv2.putText(
                    vis_image, f"res={rs:.1f}", (cx + 4, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 120), 1, cv2.LINE_AA
                )

    hand_centroids = []
    obj_centroids = []
    for oid, g in geom.items():
        c = g.get("centroid")
        if c is None:
            continue
        if oid < 100:
            hand_centroids.append((int(oid), float(c[0]), float(c[1])))
        else:
            obj_centroids.append((int(oid), float(c[0]), float(c[1])))
    min_pair = None
    min_dist = None
    for a in range(len(obj_centroids)):
        for b in range(a + 1, len(obj_centroids)):
            _, ax, ay = obj_centroids[a]
            _, bx, by = obj_centroids[b]
            d = float(np.hypot(ax - bx, ay - by))
            if min_dist is None or d < min_dist:
                min_dist = d
                min_pair = (obj_centroids[a][0], obj_centroids[b][0])

    y0 = 20
    cv2.putText(
        vis_image,
        f"bg_motion={'ok' if bg_ok else 'fail'} | bg_mag={bg_mag:.2f}",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    y0 += 18
    if min_pair is not None and min_dist is not None:
        cv2.putText(
            vis_image,
            f"min_obj_dist obj-{min_pair[0]-100} <-> obj-{min_pair[1]-100}: {min_dist:.1f}",
            (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    if bg_vec is not None:
        p0 = (10, y0 + 20)
        p1 = (int(p0[0] + bg_vec[0]), int(p0[1] + bg_vec[1]))
        cv2.arrowedLine(vis_image, p0, p1, (0, 140, 255), 2, tipLength=0.3)
        cv2.putText(
            vis_image,
            "bg_vec",
            (p1[0] + 4, p1[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 90, 160),
            1,
            cv2.LINE_AA,
        )
    for hid, hx, hy in hand_centroids:
        nearest = None
        nearest_dist = None
        for oid, ox, oy in obj_centroids:
            d = float(np.hypot(hx - ox, hy - oy))
            if nearest_dist is None or d < nearest_dist:
                nearest_dist = d
                nearest = (oid, ox, oy)
        if nearest is None:
            continue
        oid, ox, oy = nearest
        cv2.line(vis_image, (int(hx), int(hy)), (int(ox), int(oy)), (60, 180, 60), 1)
        cv2.putText(
            vis_image,
            f"h{hid}->o{oid-100} d={nearest_dist:.1f}",
            (int(hx) + 4, int(hy) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 120, 40),
            1,
            cv2.LINE_AA,
        )
    return geom
