"""
ONNX-based WILOR hand tracker (ported from the HybridOffloading WILOR ONNX
webcam demo).

The heavy WILOR PyTorch model is replaced by an exported ONNX graph, so only a
YOLO hand detector (ultralytics) plus onnxruntime are needed at inference time.
All WILOR helper functions live in the dependency-light ``wilor_onnx_utils``
module next to this file.
"""
import os

import cv2
import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError(
        'onnxruntime is required for the ONNX hand tracker. '
        'Install it with: pip install onnxruntime-gpu'
    ) from exc

from ultralytics import YOLO

from .wilor_onnx_utils import (
    get_config,
    expand_to_aspect_ratio,
    generate_image_patch_cv2,
    convert_cvimg_to_tensor,
    cam_crop_to_full,
)


# --- Default model asset paths (repo-root: pretrained/handtracker_onnx/) ------
# Weights are NOT tracked in git (see WEIGHTS.md / .gitignore); place them at
# the paths below or override via the constructor arguments.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_WEIGHTS_DIR = os.path.join(_REPO_ROOT, 'pretrained', 'handtracker_onnx')
DEFAULT_ONNX_PATH = os.path.join(_WEIGHTS_DIR, 'wilor_final_standard.onnx')
DEFAULT_CFG_PATH = os.path.join(_WEIGHTS_DIR, 'model_config.yaml')
DEFAULT_MANO_PATH = os.path.join(_WEIGHTS_DIR, 'mano_data')
DEFAULT_DETECTOR_PATH = os.path.join(_WEIGHTS_DIR, 'detector.pt')


def load_wilor_cfg(cfg_path, mano_path):
    cfg = get_config(cfg_path, update_cachedir=True)
    if ('vit' in cfg.MODEL.BACKBONE.TYPE) and ('BBOX_SHAPE' not in cfg.MODEL):
        cfg.defrost()
        assert cfg.MODEL.IMAGE_SIZE == 256, f'MODEL.IMAGE_SIZE ({cfg.MODEL.IMAGE_SIZE}) must be 256'
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.freeze()

    if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
        cfg.defrost()
        cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        cfg.freeze()

    if ('DATA_DIR' in cfg.MANO) or ('MODEL_PATH' in cfg.MANO) or ('MEAN_PARAMS' in cfg.MANO):
        cfg.defrost()
        cfg.MANO.DATA_DIR = mano_path
        cfg.MANO.MODEL_PATH = mano_path
        cfg.MANO.MEAN_PARAMS = os.path.join(mano_path, 'mano_mean_params.npz')
        cfg.freeze()

    return cfg


def select_onnx_providers():
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        print('Attempting to use ONNX Runtime GPU provider: CUDAExecutionProvider')
        return ['CUDAExecutionProvider']
    if 'TensorrtExecutionProvider' in available:
        print('Attempting to use ONNX Runtime GPU provider: TensorrtExecutionProvider')
        return ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
    print('Using ONNX Runtime CPU provider: CPUExecutionProvider')
    return ['CPUExecutionProvider']


class WilorHandTrackerONNX:
    """WILOR hand tracker running the exported ONNX graph.

    Parameters
    ----------
    onnx_path, cfg_path, mano_path, detector_path : str
        Paths to the model assets. Default to ``pretrained/handtracker_onnx/``
        (repo-root relative); override to point elsewhere.
    det_conf : float
        YOLO detection confidence threshold.
    rescale_factor : float
        Bounding-box rescale factor used when cropping the hand patch.
    """

    def __init__(self,
                 onnx_path=DEFAULT_ONNX_PATH,
                 cfg_path=DEFAULT_CFG_PATH,
                 mano_path=DEFAULT_MANO_PATH,
                 detector_path=DEFAULT_DETECTOR_PATH,
                 det_conf=0.3,
                 rescale_factor=2.0):
        for label, path in [('ONNX model', onnx_path), ('config', cfg_path),
                            ('detector', detector_path)]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'{label} not found: {path}')

        self.cfg = load_wilor_cfg(cfg_path, mano_path)
        self.img_size = int(self.cfg.MODEL.IMAGE_SIZE)
        self.det_conf = det_conf
        self.rescale_factor = rescale_factor

        providers = select_onnx_providers()
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as exc:
            print('Warning: failed to initialize ONNX Runtime with providers', providers)
            print(exc)
            print('Falling back to CPUExecutionProvider.')
            self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.providers = self.session.get_providers()

        self.detector = YOLO(detector_path)

    # -- internal helpers -----------------------------------------------------
    def _preprocess_hand_patch(self, image, box, do_flip=False):
        # box: [x1, y1, x2, y2]
        # do_flip: WILOR/MANO 는 오른손 전용 모델이다. 왼손은 패치를 좌우 반전해 넣고
        #   출력의 x 를 다시 뒤집어야 한다 (process() 의 is_right==0 분기). 반전 없이
        #   넣으면 왼손 결과가 무의미해진다.
        center = (box[2:4] + box[0:2]) / 2.0
        scale = self.rescale_factor * (box[2:4] - box[0:2]) / 200.0
        bbox_size = expand_to_aspect_ratio(scale * 200.0,
                                           target_aspect_ratio=self.cfg.MODEL.BBOX_SHAPE).max()

        cvimg = image.copy()
        img_patch_cv, _ = generate_image_patch_cv2(
            cvimg,
            float(center[0]), float(center[1]),
            float(bbox_size), float(bbox_size),
            self.img_size, self.img_size,
            do_flip, 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT,
        )

        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)
        mean = 255.0 * np.array(self.cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
        std = 255.0 * np.array(self.cfg.MODEL.IMAGE_STD, dtype=np.float32)
        for c in range(min(img_patch.shape[0], 3)):
            img_patch[c] = (img_patch[c] - mean[c]) / std[c]

        return img_patch[np.newaxis, :, :, :].astype(np.float32), center, bbox_size

    @staticmethod
    def _project_full_img(points, cam_trans, focal_length, img_res):
        camera_center = np.array([img_res[0] / 2.0, img_res[1] / 2.0], dtype=np.float32)
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[0, 2] = camera_center[0]
        K[1, 2] = camera_center[1]

        points = points + cam_trans
        points = points / points[..., -1:]  # perspective divide
        projected = (K @ points.T).T
        return projected[..., :2]

    # -- public API -----------------------------------------------------------
    def warmup(self, image=None):
        """Run one dummy forward pass to trigger ONNX/CUDA kernel init and YOLO
        warmup so the first real frame does not pay the (~1s) first-run cost.

        A blank dummy image yields no detection, so the expensive ViT ONNX path
        is exercised directly with a dummy input patch in addition to warming
        the detector.
        """
        if image is None:
            image = np.zeros((360, 640, 3), dtype=np.uint8)
        self.detect(image)
        dummy_patch = np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy_patch})

    def detect(self, frame):
        """Run the YOLO detector and return (bboxes[N,4], classes[N]) or (None, None)."""
        detections = self.detector.predict(frame, conf=self.det_conf, verbose=False, max_det=5)[0]
        if len(detections) == 0:
            return None, None

        bboxes, classes = [], []
        for det in detections:
            Bbox = det.boxes.data.cpu().detach().numpy().squeeze()
            cls = int(det.boxes.cls.cpu().detach().numpy().squeeze())
            bboxes.append(Bbox[:4].astype(np.float32))
            classes.append(cls)
        if len(bboxes) == 0:
            return None, None
        return np.stack(bboxes), np.array(classes, dtype=np.int32)

    def process(self, frame):
        """Estimate hand pose for a single BGR frame.

        Returns a dict with keys:
            joints_2d  : (21, 2) pixel coordinates in the frame
            joints_3d  : (21, 3) wrist-relative 3D joints (metres)
            vertices   : (778, 3) MANO vertices
            bbox       : (4,) [x1, y1, x2, y2] of the chosen hand
            is_right   : int (1 right hand, 0 left hand)
        or ``None`` when no hand is detected.
        """
        bboxes, classes = self.detect(frame)
        if bboxes is None:
            return None

        right_indices = np.where(classes == 1)[0]
        chosen_idx = right_indices[0] if right_indices.size > 0 else 0
        box = bboxes[chosen_idx]
        is_right = int(classes[chosen_idx])

        patch, center, bbox_size = self._preprocess_hand_patch(frame, box, do_flip=(is_right == 0))

        outputs = self.session.run(None, {self.input_name: patch})
        pred_cam = outputs[0].astype(np.float32)
        pred_joints = outputs[1].astype(np.float32)[0]
        pred_vertices = outputs[2].astype(np.float32)[0]

        multiplier = float(2 * is_right - 1)
        pred_cam[0, 1] = pred_cam[0, 1] * multiplier

        scaled_focal_length = float(
            self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * max(frame.shape[1], frame.shape[0]))
        box_center = np.array([center], dtype=np.float32)
        box_size = np.array([bbox_size], dtype=np.float32)
        img_size_np = np.array([[frame.shape[1], frame.shape[0]]], dtype=np.float32)
        focal_length = torch.tensor(scaled_focal_length, dtype=torch.float32)

        cam_t_full = cam_crop_to_full(
            torch.from_numpy(pred_cam),
            torch.from_numpy(box_center),
            torch.from_numpy(box_size),
            torch.from_numpy(img_size_np),
            focal_length,
        ).cpu().numpy()[0]

        if is_right == 0:
            pred_joints[:, 0] = -pred_joints[:, 0]
            pred_vertices[:, 0] = -pred_vertices[:, 0]

        joints_2d = self._project_full_img(
            pred_joints, cam_t_full, scaled_focal_length,
            np.array([frame.shape[1], frame.shape[0]], dtype=np.float32))

        return {
            'joints_2d': joints_2d.astype(np.float32),
            'joints_3d': pred_joints.astype(np.float32),
            'vertices': pred_vertices.astype(np.float32),
            'bbox': box.astype(np.float32),
            'is_right': is_right,
        }
