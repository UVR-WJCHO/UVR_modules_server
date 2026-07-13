"""
Self-contained utilities extracted from the WILOR package for ONNX-based hand
tracking inference.

Only the functions actually required by the ONNX pipeline are included here so
that importing this module does NOT pull in the heavy 3D-rendering dependencies
(pyrender, trimesh, matplotlib, scikit-image) that the original ``wilor.utils``
package imports at module load time.

Sources (modules/handtracker_wilor/wilor):
  - configs/__init__.py   -> default_config, get_config
  - datasets/utils.py     -> expand_to_aspect_ratio, rotate_2d,
                             gen_trans_from_patch_cv, generate_image_patch_cv2,
                             convert_cvimg_to_tensor
  - utils/renderer.py     -> cam_crop_to_full
  - utils/visualize.py    -> draw_2d_skeleton (+ color_hand_joints)
"""
import os

import cv2
import numpy as np
import torch
from yacs.config import CfgNode as CN


# ---------------------------------------------------------------------------
# Config  (from wilor/configs/__init__.py)
# ---------------------------------------------------------------------------
CACHE_DIR_PRETRAINED = "./pretrained_models/"

_C = CN(new_allowed=True)

_C.GENERAL = CN(new_allowed=True)
_C.GENERAL.RESUME = True
_C.GENERAL.TIME_TO_RUN = 3300
_C.GENERAL.VAL_STEPS = 100
_C.GENERAL.LOG_STEPS = 100
_C.GENERAL.CHECKPOINT_STEPS = 20000
_C.GENERAL.CHECKPOINT_DIR = "checkpoints"
_C.GENERAL.SUMMARY_DIR = "tensorboard"
_C.GENERAL.NUM_GPUS = 1
_C.GENERAL.NUM_WORKERS = 4
_C.GENERAL.MIXED_PRECISION = True
_C.GENERAL.ALLOW_CUDA = True
_C.GENERAL.PIN_MEMORY = False
_C.GENERAL.DISTRIBUTED = False
_C.GENERAL.LOCAL_RANK = 0
_C.GENERAL.USE_SYNCBN = False
_C.GENERAL.WORLD_SIZE = 1

_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.WARMUP = False
_C.TRAIN.NORMALIZE_PER_IMAGE = False
_C.TRAIN.CLIP_GRAD = False
_C.TRAIN.CLIP_GRAD_VALUE = 1.0
_C.LOSS_WEIGHTS = CN(new_allowed=True)

_C.DATASETS = CN(new_allowed=True)

_C.MODEL = CN(new_allowed=True)
_C.MODEL.IMAGE_SIZE = 224

_C.EXTRA = CN(new_allowed=True)
_C.EXTRA.FOCAL_LENGTH = 5000

_C.DATASETS.CONFIG = CN(new_allowed=True)
_C.DATASETS.CONFIG.SCALE_FACTOR = 0.3
_C.DATASETS.CONFIG.ROT_FACTOR = 30
_C.DATASETS.CONFIG.TRANS_FACTOR = 0.02
_C.DATASETS.CONFIG.COLOR_SCALE = 0.2
_C.DATASETS.CONFIG.ROT_AUG_RATE = 0.6
_C.DATASETS.CONFIG.TRANS_AUG_RATE = 0.5
_C.DATASETS.CONFIG.DO_FLIP = False
_C.DATASETS.CONFIG.FLIP_AUG_RATE = 0.5
_C.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.10


def default_config():
    """Get a yacs CfgNode object with the default config values."""
    return _C.clone()


def get_config(config_file, merge=True, update_cachedir=False):
    """Read a config file and optionally merge it with the default config."""
    if merge:
        cfg = default_config()
    else:
        cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)

    if update_cachedir:
        def update_path(path):
            if os.path.isabs(path):
                return path
            return os.path.join(CACHE_DIR_PRETRAINED, path)

        cfg.MANO.MODEL_PATH = update_path(cfg.MANO.MODEL_PATH)
        cfg.MANO.MEAN_PARAMS = update_path(cfg.MANO.MEAN_PARAMS)

    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Image patch / bbox utilities  (from wilor/datasets/utils.py)
# ---------------------------------------------------------------------------
def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])


def rotate_2d(pt_2d, rot_rad):
    """Rotate a 2D point on the x-y plane."""
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height,
                            dst_width, dst_height, scale, rot):
    """Create transformation matrix for the bounding box crop."""
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def generate_image_patch_cv2(img, c_x, c_y, bb_width, bb_height,
                             patch_width, patch_height, do_flip, scale, rot,
                             border_mode=cv2.BORDER_CONSTANT, border_value=0):
    """Crop the input image and return the crop and the transformation matrix."""
    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height,
                                    patch_width, patch_height, scale, rot)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR,
                               borderMode=border_mode,
                               borderValue=border_value)
    # Force borderValue=cv2.BORDER_CONSTANT for alpha channel
    if (img.shape[2] == 4) and (border_mode != cv2.BORDER_CONSTANT):
        img_patch[:, :, 3] = cv2.warpAffine(img[:, :, 3], trans,
                                            (int(patch_width), int(patch_height)),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT)

    return img_patch, trans


def convert_cvimg_to_tensor(cvimg):
    """Convert image from HWC to CHW format."""
    # from h,w,c(OpenCV) to c,h,w
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    # from int to float
    img = img.astype(np.float32)
    return img


# ---------------------------------------------------------------------------
# Camera utilities  (from wilor/utils/renderer.py)
# ---------------------------------------------------------------------------
def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


# ---------------------------------------------------------------------------
# Visualization  (from wilor/utils/visualize.py)
# ---------------------------------------------------------------------------
color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little


def draw_2d_skeleton(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    """
    assert pose_uv.shape[0] == 21
    skeleton_overlay = image.copy()
    marker_sz = 6
    line_wd = 3
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype('int32'), pose_uv[joint_ind, 1].astype('int32')
        cv2.circle(
            skeleton_overlay, joint,
            radius=marker_sz, color=color_hand_joints[joint_ind] * np.array(255), thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype('int32'), pose_uv[root_ind, 1].astype('int32')
            cv2.line(
                skeleton_overlay, root_joint, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        else:
            joint_2 = pose_uv[joint_ind - 1, 0].astype('int32'), pose_uv[joint_ind - 1, 1].astype('int32')
            cv2.line(
                skeleton_overlay, joint_2, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    return skeleton_overlay
