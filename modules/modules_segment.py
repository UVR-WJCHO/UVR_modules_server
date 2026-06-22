import os, sys
import cv2
import numpy as np
from typing import Optional, Tuple
import mediapipe as mp
import torch

sys.path.append('./modules/segmentor')
from sam2_realtime.sam2.build_sam import build_sam2_realtime_predictor
import time
import ultralytics
import json
from pycocotools import mask as mask_utils
import math
# import pyrealsense2 as rs
import glob
from segmentor.hotrack_utils import *


class HOSegmentor():
    def __init__(self):
        # MediaPipe Hands 초기화
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(max_num_hands=10, model_complexity=0)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # SAM2.1 초기화
        checkpoint = "./modules/segmentor/sam2_realtime/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"
        self.tracker = build_sam2_realtime_predictor(model_cfg, checkpoint, device=self.device)

        self.yolo_hand_model = ultralytics.YOLO("./modules/segmentor/100DOH_small.pt", verbose=False)

        self.all_masks = None
        self.all_ids = []
        self.inference_state = {}
        self.all_colors = {}  # 각 객체별 고유 색상 저장
        # 상수
        self.MAX_TRACK_OBJECTS = 5  # 오브젝트 트래킹 갯수 설정

    def get_hand_landmarks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 변환
        results = self.hands.process(rgb_image)  # MediaPipe Hands로 처리
        multi_hand_landmarks = results.multi_hand_landmarks
        if multi_hand_landmarks:
            return multi_hand_landmarks  # 여러 손 랜드마크 반환
        return None

    def find_object_numpy(self,
                          depth_image, bounding_box, hand_bounding_box, target_depth,
                          interaction_vector, origin, alpha=0.8, bbox_distance_ratio=1):

        min_x, min_y, max_x, max_y = bounding_box
        # h, w = depth_image.shape

        # 손 bbox의 대각선 길이를 기준으로 max_distance 계산
        h_min_x, h_min_y, h_max_x, h_max_y = hand_bounding_box
        hand_diag = math.sqrt((h_max_x - h_min_x) ** 2 + (h_max_y - h_min_y) ** 2)
        max_distance = hand_diag * bbox_distance_ratio

        # 손 마스크 제거: all_masks가 전역 변수라고 가정
        depth_no_hand = depth_image.copy()
        all_mask_dilated = dilate_mask_numpy(self.all_masks, kernel_size=21)
        depth_no_hand[all_mask_dilated == 1] = 0
        depth_no_hand[np.isnan(depth_no_hand)] = 0
        # 깊이 차이 기반 마스크 생성
        depth_diff = depth_no_hand - target_depth
        mask_depth = (depth_diff == 0).astype(np.uint8)  # ((depth_diff > -1) & (depth_diff < 1)).astype(np.uint8)
        # 바운딩 박스 외 영역은 0으로 처리
        mask_depth[:min_y, :] = 0
        mask_depth[max_y:, :] = 0
        mask_depth[:, :min_x] = 0
        mask_depth[:, max_x:] = 0
        # 후보점 찾기 (벡터화: np.argwhere로 모든 후보 추출)
        candidate_points = np.argwhere(mask_depth == 1)  # (row, col) 형태
        if candidate_points.shape[0] == 0:
            return None

        # (x, y) 좌표 교환: point_xy = [col, row]
        points_xy = candidate_points[:, ::-1].astype(np.float32)
        origin = np.array(origin, dtype=np.float32)

        # 모든 점에 대한 벡터, 거리 계산
        vectors = points_xy - origin  # shape (N,2)
        distances = np.linalg.norm(vectors, axis=1)

        # max_distance 초과 점 제거
        valid_mask = distances <= max_distance
        if not np.any(valid_mask):
            return None
        points_xy = points_xy[valid_mask]
        vectors = vectors[valid_mask]
        distances = distances[valid_mask]

        # 각 후보의 방향 (unit vector) 계산
        directions = vectors / (distances[:, None] + 1e-6)
        # interaction_vector와 내적 (유사도) 계산
        dot_prods = np.dot(directions, interaction_vector)
        scores = alpha * dot_prods - (1 - alpha) * (distances / max_distance)

        best_idx = np.argmax(scores)
        best_point = points_xy[best_idx].astype(int).tolist()

        return best_point

    def get_hotrack_datas(self, color_image, depth_transformed_image):

        if sum(1 for id in self.all_ids if id >= 100) > self.MAX_TRACK_OBJECTS:
            return []

        height, width = color_image.shape[:2]

        # (1) YOLO 결과 추출 (모델은 GPU inference 후 결과만 CPU로 전달)
        st = time.time()
        yolo_results = self.yolo_hand_model(color_image, verbose=False)
        yolo_bbs = []
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                yolo_bbs.append([cls, x1, y1, x2, y2])
        if not yolo_bbs:
            return []
        # et = time.time()
        # print("yolo inference time:", et - st)
        # st = time.time()
        # (2) mediapipe 결과 (한 번만 실행)
        multi_hand_landmarks = self.get_hand_landmarks(color_image)
        if not multi_hand_landmarks:
            return []
        # et = time.time()
        # print("mediapipe inference time:", et - st)
        # mediapipe 손 바운딩 박스 계산 (전체 landmark vectorized 처리)
        hand_boxes = []
        for hand_landmarks in multi_hand_landmarks:
            positions = get_unnormalized_positions(hand_landmarks.landmark, width, height)
            hb = [int(positions[:, 0].min()), int(positions[:, 1].min()),
                  int(positions[:, 0].max()), int(positions[:, 1].max())]
            hand_boxes.append(hb)
        hand_boxes = np.array(hand_boxes)  # shape: (m,4)

        # YOLO 박스 배열 구성 및 클래스 분리
        yolo_bbs = np.array(yolo_bbs)  # shape: (n,5)
        yolo_classes = yolo_bbs[:, 0].astype(int)
        bbox_array = yolo_bbs[:, 1:].astype(np.int32)  # shape: (n,4)

        # (3) 각 손 박스와 YOLO 박스 간의 교집합 비율 계산 (벡터화)
        overlap_ratio_matrix = calculate_overlap_ratio_np(hand_boxes, bbox_array)  # shape: (m, n)

        # 유효한 (손, YOLO) 쌍 추출 (threshold 이상인 경우만)
        threshold = 0.85
        valid_indices = np.argwhere(overlap_ratio_matrix >= threshold)
        if valid_indices.size == 0:
            return []
        valid_ratios = overlap_ratio_matrix[overlap_ratio_matrix >= threshold]
        sort_order = np.argsort(-valid_ratios)  # 내림차순 정렬
        sorted_pairs = valid_indices[sort_order]

        used_hand_ids = set()
        used_yolo_ids = set()
        hotrack_data = []

        for pair in sorted_pairs:
            i, j = pair
            if i in used_hand_ids or j in used_yolo_ids:
                continue
            used_hand_ids.add(i)
            used_yolo_ids.add(j)

            # (4) 매칭된 박스 및 교집합 박스 계산
            hx1, hy1, hx2, hy2 = hand_boxes[i]
            yx1, yy1, yx2, yy2 = bbox_array[j]
            ix1 = max(hx1, yx1)
            iy1 = max(hy1, yy1)
            ix2 = min(hx2, yx2)
            iy2 = min(hy2, yy2)

            # (5) mediapipe 손 landmark 벡터화 처리
            hand_landmarks_obj = multi_hand_landmarks[i]
            hand_screen = get_unnormalized_positions(hand_landmarks_obj.landmark, width, height)
            hand_screen = hand_screen.tolist()  # list 변환

            # (6) 각 landmark의 depth 값을 depth_transformed_image에서 읽기
            hand_screen_depths = []
            for lm in hand_landmarks_obj.landmark:
                cx, cy = get_unnormalized_position(lm, width, height)
                hand_screen_depths.append(depth_transformed_image[cy, cx])

            # (7) 손의 0번 landmark와 각 손끝(fingertip) landmark 간 벡터 계산 후 평균
            fingertip_indices = [4, 8, 12, 16, 20]
            base = np.array(hand_screen[0])
            vectors = [np.array(hand_screen[idx]) - base for idx in fingertip_indices]
            mean_vector = np.mean(vectors, axis=0)
            norm = np.linalg.norm(mean_vector)
            if norm != 0:
                mean_vector = mean_vector / norm

            # (8) YOLO 박스의 네 꼭짓점 중, 0번 landmark와 방향 유사도가 가장 높은 꼭짓점 선택
            yolo_corners = np.array([[yx1, yy1], [yx2, yy1], [yx1, yy2], [yx2, yy2]])
            yolo_vectors = yolo_corners - base
            similarities = np.dot(yolo_vectors, mean_vector)
            best_corner = yolo_corners[int(np.argmax(similarities))]
            intersect_object_box = [
                int(min(base[0], best_corner[0])),
                int(min(base[1], best_corner[1])),
                int(max(base[0], best_corner[0])),
                int(max(base[1], best_corner[1]))
            ]

            record = {
                'class': int(yolo_classes[j]),
                'yolo_box': [yx1, yy1, yx2, yy2],
                'intersect_box': [ix1, iy1, ix2, iy2],
                'object_box': intersect_object_box,
                'interaction_vector': mean_vector,
                'hand_screen': hand_screen,
                'hand_screen_depths': hand_screen_depths,
            }
            hotrack_data.append(record)

        return hotrack_data

    @torch.inference_mode()
    def run(self, color_image, depth_image, hotrack_datas, is_first_call, frame_idx):
        height, width = color_image.shape[:2]

        tracking_mask_dic = {}
        added_hand_count = 0
        added_obj_count = 0

        # 첫 프레임 로드/ 다음 프레임 추가
        if is_first_call:
            self.inference_state = self.tracker.init_state(color_image)
        else:
            self.inference_state = self.tracker.add_frame(self.inference_state, color_image)

        # 트래킹할 수 있는 오브젝트가 있다면 트래킹을 하여 all_mask 최신화
        self.all_masks = np.zeros((height, width), dtype=np.uint8)
        if len(self.all_ids) > 0:
            frame_idx, self.all_ids, out_mask_logits, self.inference_state = self.tracker.get_mask(self.inference_state,
                                                                                                   frame_idx)
            if out_mask_logits is not None:
                for idx, all_id in enumerate(self.all_ids):
                    out_mask = get_each_mask(out_mask_logits, idx, height, width)
                    tracking_mask_dic[str(all_id)] = out_mask
                    self.all_masks = cv2.bitwise_or(self.all_masks, out_mask)

        # 손, 객체 프롬프트를 hotrack_data 마다 추가
        if len(hotrack_datas) > 0:
            for hotrack_data in hotrack_datas:
                # 손 추가
                hand_prompts = [hotrack_data['hand_screen'][i] for i in [0, 1, 5, 9, 13, 17]]
                hand_skipped = is_prompt_in_mask(self.all_masks, hand_prompts)
                if not hand_skipped:
                    start_idx = len(self.all_ids)
                    frame_idx, self.all_ids, out_mask_logits, self.inference_state = self.tracker.add_new_points_or_box(
                        self.inference_state, frame_idx=frame_idx, obj_id=start_idx, points=hand_prompts,
                        labels=np.ones(len(hand_prompts)))
                    # cp = color_image.copy()
                    # for point in hand_prompts:
                    #    cv2.circle(cp, point, 5, (0, 255, 0), -1)
                    # cv2.imshow("hand", cp)
                    out_mask = get_each_mask(out_mask_logits, len(self.all_ids) - 1, height, width)
                    tracking_mask_dic[str(start_idx)] = out_mask
                    self.all_masks = cv2.bitwise_or(self.all_masks, out_mask)
                    added_hand_count += 1
                # 객체 추가
                if (hotrack_data['class'] == 1):
                    hand_depths = np.array(hotrack_data['hand_screen_depths'])[4:21:4]
                    mask = depth_image > sorted(hand_depths)[-2]
                    # mask = depth_image < sorted(hand_depths)[0]
                    depth_image[mask] = 0

                    obj_point = self.find_object_numpy(depth_image, hotrack_data['object_box'],
                                                       hotrack_data['intersect_box'], np.median(hand_depths),
                                                       hotrack_data['interaction_vector'],
                                                       hotrack_data['hand_screen'][0])

                    if obj_point is not None:
                        if not is_prompt_in_mask(self.all_masks, [obj_point]):
                            start_idx = 100 + len(self.all_ids)
                            frame_idx, self.all_ids, out_mask_logits, self.inference_state = self.tracker.add_new_points_or_box(
                                self.inference_state, frame_idx=frame_idx, obj_id=start_idx, points=[obj_point],
                                labels=np.ones([1]))
                            out_mask = get_each_mask(out_mask_logits, len(self.all_ids) - 1, height, width)
                            tracking_mask_dic[str(start_idx)] = out_mask
                            self.all_masks = cv2.bitwise_or(self.all_masks, out_mask)

                            cv2.circle(color_image, obj_point, 5, 1000, -1)
                            added_obj_count += 1

        # 지워야 할 마스크 탐색
        # 총 추가된 마스크 수
        added_all_count = added_hand_count + added_obj_count
        # 제거할 마스크 탐색 후 제거
        if added_all_count > 0:
            # 제거할 마스크 탐색
            id_to_removes = []
            for i in range(len(self.all_ids) - 1, -1, -1):
                if i < len(self.all_ids) - added_all_count:
                    break
                target_mask = tracking_mask_dic[str(self.all_ids[i])]
                if is_invalid_mask(target_mask, self.all_ids[i], height, width):
                    id_to_removes.append(self.all_ids[i])
                    continue
                for j in range(i - 1, -1, -1):
                    # 새로운 마스크
                    # if j > len(self.all_ids) - added_all_count - 1:
                    earlier_mask = tracking_mask_dic[str(self.all_ids[j])]
                    if is_same_object(target_mask, earlier_mask):
                        if self.all_ids[i] not in id_to_removes:
                            id_to_removes.append(self.all_ids[i])
                        else:
                            break
                    # 기존 마스크
                    # else:
                    #    if is_overlay_mask(target_mask, origin_masks): # 추가된 마스크가 기존의 마스크들과 겹친다면
                    #        if self.all_ids[i] not in id_to_removes: # 제거할 마스크배열에 중복으로 추가되었는지 확인후 추가
                    #            id_to_removes.append(self.all_ids[i])
                    #    break
            # 마스크 제거
            for sam_id in id_to_removes:
                tracking_mask_dic[str(sam_id)] = None
                # print(f"Removing invalid mask for ID {sam_id}")
                self.all_ids, _ = self.tracker.remove_object(self.inference_state, sam_id, True)

        '''
        시각화 부분
        0~99 : 손
        100~ : 객체
        마스크를 원한다면 여기서 tracking_mask_dic 사용하여 뽑을 수 있음.
        '''
        result_masks = []
        for sam_id in tracking_mask_dic.keys():
            out_mask = tracking_mask_dic[sam_id]
            if out_mask is None:
                continue

            if int(sam_id) > 99:
                result_masks.append(out_mask)

            # process_segmentation_COCO(frame_idx, out_mask, sam_id)
            overlay = color_image.copy()
            if sam_id not in self.all_colors:
                self.all_colors[sam_id] = tuple(np.random.randint(0, 255, 3).tolist())
            color = self.all_colors[sam_id]
            overlay[out_mask == 1] = color
            x, y, w, h = cv2.boundingRect(out_mask)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            cv2.putText(overlay, sam_id, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            color_image = cv2.addWeighted(overlay, 0.5, color_image, 0.5, 0, color_image)

        return color_image, result_masks  # 시각화된 이미지 반환


def main():
    rgb_path = "./modules/segmentor/metaobj_modeling_example_2/01_rgb"
    depth_path = "./modules/segmentor/metaobj_modeling_example_2/04_pred_depth"
    rgb_files = sorted(glob.glob(os.path.join(rgb_path, "frame_*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_path, "frame_*.png")))

    is_first_call = True
    frame_idx = 0
    total_process_times = []
    total_process_memories = []

    hosegmentor = HOSegmentor()

    # 프레임 읽기
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        try:
            for rgb_file, depth_file in zip(rgb_files, depth_files):
                rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)  # RGB 이미지
                # depth_image = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)  # 깊이 이미지
                depth_image = read_depth_hw(depth_file)
                height, width = rgb_image.shape[:2]
                h_color_image = rgb_image.copy()

                starttime = time.time()
                torch.cuda.reset_peak_memory_stats()

                hotrack_datas = hosegmentor.get_hotrack_datas(h_color_image, depth_image)

                h_color_image, result_masks = hosegmentor.run(h_color_image, depth_image, width, height, hotrack_datas, is_first_call,
                                                frame_idx)

                print("all_ids ", hosegmentor.all_ids)

                frame_idx += 1
                is_first_call = False
                cv2.imshow("Segmentation", h_color_image)

                for i, mask in enumerate(result_masks):
                    cv2.imshow(f"mask {i}", mask*255)

                key = cv2.waitKey(0)
                if key == ord("q"):  # 'q' 키로 종료\
                    break

        finally:
            # pipeline.stop()
            pass


if __name__ == "__main__":
    main()