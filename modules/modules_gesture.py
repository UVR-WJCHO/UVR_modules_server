import os
import numpy as np
from collections import deque
import torch
# from handtracker.module_SARTE import HandTracker
# from handtracker_wilor.module_WILOR import HandTracker_wilor
from gestureclassifier.model_update import create_model  # Assumed external module for model loading


class GestureClassfier:
    """Sequence-based gesture classifier."""

    def __init__(self, ckpt, seq_len=16, model_opt=1):
        if model_opt == 0 or model_opt >= 2 and model_opt < 6:
            num_feature = 78
            self.flag_partial = False
        else:
            num_feature = 60
            self.flag_partial = True

        self.seq_len = seq_len
        self.model = create_model(num_features=num_feature, num_classes=15, model_opt=model_opt)  # Loads model architecture

        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['model_state_dict']

        # "module." prefix가 있는지 확인
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        # prefix 제거
        if has_module_prefix:
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict

        self.model.load_state_dict(new_state_dict)

        self.model.eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        # Normalization constants (Must be predetermined from the dataset)
        self.norm_ratio_x = 100
        self.norm_ratio_y = 100
        self.norm_ratio_z = 100

        self.partial_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 20]
        self.seq_len = seq_len
        self.idx_to_class = {
            0: 'CClock_index', 1: 'CClock_thumb',
            2: 'Clock_index', 3: 'Clock_thumb',
            4: 'Down_index', 5: 'Down_thumb',
            6: 'Left_index', 7: 'Left_thumb',
            8: 'Natural',
            9: 'Right_index', 10: 'Right_thumb',
            11: 'Tap_index', 12: 'Tap_thumb',
            13: 'Up_index', 14: 'Up_thumb'
        }



    def _extract_partialhand(self, pts_norm):
        """Extract features for partial hand joints."""
        pts_norm = np.asarray(pts_norm)
        pts_norm_part = []
        for frame_idx in range(pts_norm.shape[0]):
            target_pose = pts_norm[frame_idx, :63].reshape(21, 3)
            target_angle = pts_norm[frame_idx, 63:]

            target_pose = target_pose[self.partial_idx, :]
            target_pose = target_pose.flatten()

            pts_ = np.concatenate((target_pose, target_angle), axis=0)
            pts_norm_part.append(pts_)

        return np.array(pts_norm_part)

    def _compute_ang_from_joint(self, joint):
        """Compute angles between joints (21, 3)."""
        # Define parent and child joint indices
        v1_indices = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
        v2_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        v1 = joint[v1_indices, :]
        v2 = joint[v2_indices, :]
        v = v2 - v1

        # Normalize vectors
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Calculate angles using arccos of dot product
        # Indices for angle calculation pairs
        angle_pairs_v1 = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        angle_pairs_v2 = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

        dot_product = np.einsum('nt,nt->n', v[angle_pairs_v1, :], v[angle_pairs_v2, :])
        angle = np.arccos(dot_product)
        angle = np.degrees(angle)

        return angle
    
    def _normalize_and_combine(self, frame_data, frame_idx):
        """
        Normalizes the hand pose and combines it with angle data for sequence input.
        frame_data: [pose_flatten (63)] + [angle_label (15)] -> size 78
        """
        target_pose = frame_data[:63].reshape((21, 3))
        target_angle = frame_data[63:]

        # Root joint subtraction for translation invariance
        root_pose = target_pose[0, :]
        norm_pose = target_pose - root_pose

        # Apply ratio normalization for scale invariance
        norm_pose[:, 0] = norm_pose[:, 0] / self.norm_ratio_x
        norm_pose[:, 1] = norm_pose[:, 1] / self.norm_ratio_y
        norm_pose[:, 2] = norm_pose[:, 2] / self.norm_ratio_z

        # Combine normalized pose and normalized angle (angle normalized to [-1, 1])
        norm_data = np.concatenate([norm_pose.flatten(), target_angle / 180.0])

        return norm_data

    def run(self, sequence_queue):
        """
        Runs the gesture classification on a sequence of hand data.

        Args:
            sequence_queue (deque): A deque of raw hand data (pose + angles).

        Returns:
            tuple: (gesture_index, gesture_label)
        """
        if len(sequence_queue) < self.seq_len:
            return -1, "Incomplete_Sequence"

        # Prepare and normalize sequence data
        input_data = [self._normalize_and_combine(raw_frame, i) for i, raw_frame in enumerate(sequence_queue)]
        if self.flag_partial:
            input_data = self._extract_partialhand(input_data)

        # Convert to tensor and run model
        input_tensor = torch.tensor(np.array(input_data), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        pred = output.argmax(1).cpu().numpy()
        gesture = self.idx_to_class[pred[0]]

        return pred[0], gesture