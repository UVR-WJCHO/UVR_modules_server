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

        # Placeholder/Example gesture labels (Must match training data)
        self.labels = ['Natural', 'Pinch', 'Grab', 'Swipe', 'Push', 'Raise']

        # Normalization constants (Must be predetermined from the dataset)
        self.norm_ratio_x = 100
        self.norm_ratio_y = 100
        self.norm_ratio_z = 100

    def _compute_ang_from_joint(self, joint_3d):
        """
        Computes joint angles from the 3D joint coordinates (Placeholder).
        (The actual implementation is assumed to be correct based on the external module).
        """
        # Assuming joint_3d is (21, 3) and returns a fixed-size array (e.g., 15)
        # The logic is abstracted as it relies on specific external libraries or mathematical formulation.
        return np.ones((15))  # Returns a dummy angle array for structure

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

        # Convert to tensor and run model
        input_tensor = torch.tensor(np.array(input_data), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        pred_idx = torch.argmax(output.squeeze(0)).item()

        if pred_idx < len(self.labels):
            gesture = self.labels[pred_idx]
            return pred_idx, gesture
        else:
            return -1, "Unknown"