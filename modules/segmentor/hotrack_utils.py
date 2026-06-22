import numpy as np
import cv2


def calculate_overlap_ratio_np(hand_boxes, bbox_array):
    """
    모든 mediapipe 손 바운딩 박스와 YOLO 박스 간의 교집합 비율을 벡터화하여 계산.
    hand_boxes: (m, 4), bbox_array: (n, 4)
    반환: (m, n) 배열
    """
    hand_x1 = hand_boxes[:, None, 0]
    hand_y1 = hand_boxes[:, None, 1]
    hand_x2 = hand_boxes[:, None, 2]
    hand_y2 = hand_boxes[:, None, 3]

    bbox_x1 = bbox_array[:, 0]
    bbox_y1 = bbox_array[:, 1]
    bbox_x2 = bbox_array[:, 2]
    bbox_y2 = bbox_array[:, 3]

    inter_x1 = np.maximum(hand_x1, bbox_x1)
    inter_y1 = np.maximum(hand_y1, bbox_y1)
    inter_x2 = np.minimum(hand_x2, bbox_x2)
    inter_y2 = np.minimum(hand_y2, bbox_y2)

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    hand_area = (hand_x2 - hand_x1) * (hand_y2 - hand_y1)
    # broadcasting을 이용하여 area 확장
    hand_area_expanded = hand_area + np.zeros((1, bbox_array.shape[0]))
    overlap_ratio = inter_area / hand_area_expanded
    overlap_ratio[hand_area_expanded == 0] = 0
    return overlap_ratio


def draw_boxes(image, hotrack_datas):
    for idx, data in enumerate(hotrack_datas):
        # YOLO 박스
        x1, y1, x2, y2 = data['yolo_box']
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, "YOLO-" + str(idx), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)

        # Intersect 박스
        x1, y1, x2, y2 = data['intersect_box']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "Intersect-" + str(idx), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)

        # Object 박스
        x1, y1, x2, y2 = data['object_box']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, "Object-" + str(idx), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
    return image

def is_same_object(target_mask, tracked_mask):
    """두 마스크 간 IoU가 1% 이상이면 같은 객체로 판단."""
    intersection = np.sum(cv2.bitwise_and(target_mask, tracked_mask))
    total = np.sum(cv2.bitwise_or(target_mask, tracked_mask))
    iou = intersection / total if total > 0 else 0
    return iou > 0.01


def is_invalid_mask(mask: np.ndarray, id, height: int, width: int,
                       min_area_ratio: float = 0.001, max_area_ratio: float = 0.25) -> bool:
    """
    마스크의 면적 비율이 너무 작거나 너무 크면 invalid로 판단.
    """
    mask_area = np.sum(mask)

    area_ratio = mask_area / (height * width)
    print("Mask area:", mask_area, "total pixels:", (height * width))
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return True
    return False

def is_prompt_in_mask(all_masks, target_prompts):
    """
    좌표 리스트(target_prompts)가 전역 마스크(all_masks)에서 1인 픽셀을 포함하는지 벡터화하여 판단.
    """
    if target_prompts is None:
        return False
    target_prompts = np.array(target_prompts)
    if all_masks is not None and target_prompts.size > 0:
        # target_prompts 배열은 각 행이 [x, y] 순서라고 가정 (이미 코드에서는 point[1]이 row)
        mask_values = all_masks[target_prompts[:, 1], target_prompts[:, 0]]
        return np.any(mask_values == 1)
    return False

def is_overlay_mask(target_mask, origin_mask):
    """두 마스크간 교차 영역 픽셀 수가 100개 이상이면 overlay로 간주."""
    all_intersected = cv2.bitwise_and(target_mask, origin_mask)
    return np.sum(all_intersected) > 100

def get_each_mask(out_mask_logits, target_obj_id, height, width):
    """
    특정 객체(target_obj_id)에 해당하는 logits로부터 마스크를 생성.
    tensor에서 CPU -> numpy 변환 후 OpenCV 비트 연산 사용.
    """
    out_mask = (out_mask_logits[target_obj_id] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # mask 초기값은 0이므로 bitwise_or는 out_mask 그대로 반환함.
    mask = cv2.bitwise_or(np.zeros((height, width), dtype=np.uint8), out_mask)
    return mask


def get_unnormalized_position(lm, width, height):
    """mediapipe 개별 landmark의 normalized 좌표를 픽셀 좌표로 변환."""
    cx = int(lm.x * width)
    cy = int(lm.y * height)
    return int(np.clip(cx, 0, width - 1)), int(np.clip(cy, 0, height - 1))


def get_unnormalized_positions(landmarks, width, height):
    """
    여러 landmark 객체 혹은 (N,2) 배열을 받아 벡터화하여 픽셀 좌표 배열로 변환.
    입력:
      landmarks: mediapipe landmark 객체 리스트 혹은 np.array([[x,y], ...])
    반환:
      (N,2) int32 배열
    """
    # landmark 객체인 경우, 객체의 x, y 속성을 추출
    if hasattr(landmarks[0], 'x'):
        coords = np.array([[lm.x, lm.y] for lm in landmarks])
    else:
        coords = np.array(landmarks)
    # 좌표 변환 및 클리핑
    positions = (coords * np.array([width, height])).astype(np.int32)
    positions = np.clip(positions, [0, 0], [width - 1, height - 1])
    return positions



def get_bounding_boxes(xs, ys):
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return [min_x, min_y, max_x, max_y]

def dilate_mask_numpy(mask, kernel_size=9):

    # 팽창 연산을 위한 필터 (모든 값이 1인 kernel 생성)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def read_depth_hw(path: str) -> np.ndarray:
    """
    어떤 포맷이 와도 (H, W) 단일 채널로 반환.
    - (H,W,3)인 그레이 PNG(시각화) → GRAY 변환
    - (H,W,1) → 채널 축 제거
    - (H,W) → 그대로
    dtype은 원본 유지(대부분 uint8 또는 uint16)
    """
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {path}")

    if depth.ndim == 3:
        # 시각화 PNG(BGR 3채널) → 단채널
        # (그레이 PNG라도 3채널로 저장된 경우가 많음)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    elif depth.ndim == 2:
        # 이미 (H,W) 단채널
        pass
    elif depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    else:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")

    return depth  # (H,W), uint8 또는 uint16