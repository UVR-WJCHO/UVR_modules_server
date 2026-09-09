"""Aria FISHEYE624 카메라 모델과 pinhole rectification.

HOT3D 의 RGB 는 어안(FISHEYE624) 이라 WiLoR 에 그대로 넣을 수 없다. WiLoR 는 원근
이미지로 학습됐다. rectification map 은 "출력 pinhole 픽셀 -> 광선 -> fisheye 투영 ->
입력 픽셀" 방향으로 만들면 역투영(반복 해법)이 필요 없다.

projectionParams(15) = [f, cx, cy, k0..k5, p0, p1, s0..s3]
`projectaria_tools` 가 없어 직접 구현했고, GT 3D joint 를 투영해 `box2d_hands.csv` 의
2D bounding box 와 대조해 검증한다(verify_projection).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_RADIAL = 6


@dataclass(frozen=True)
class Fisheye624:
    focal: float
    center: np.ndarray          # (2,)
    radial: np.ndarray          # (6,)
    tangential: np.ndarray      # (2,)
    thin_prism: np.ndarray      # (4,)
    width: int
    height: int

    @classmethod
    def from_json(cls, path: Path, label: str = "camera-rgb") -> "Fisheye624":
        models = json.loads(Path(path).read_text())
        model = next(m for m in models if m["label"] == label)
        if "FISHEYE624" not in model["projectionModelType"]:
            raise ValueError(f"{label}: 예상과 다른 모델 {model['projectionModelType']}")
        p = np.asarray(model["projectionParams"], dtype=np.float64)
        return cls(focal=float(p[0]), center=p[1:3].copy(), radial=p[3:9].copy(),
                   tangential=p[9:11].copy(), thin_prism=p[11:15].copy(),
                   width=int(model["imageWidth"]), height=int(model["imageHeight"]))

    def project(self, points: np.ndarray) -> np.ndarray:
        """(..., 3) camera-frame 3D -> (..., 2) pixel. z <= 0 인 점은 NaN."""
        pts = np.asarray(points, dtype=np.float64)
        z = pts[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = pts[..., 0] / z
            b = pts[..., 1] / z
        r = np.hypot(a, b)
        theta = np.arctan(r)

        theta_sq = theta ** 2
        radial_scale = np.ones_like(theta)
        power = theta_sq.copy()
        for k in self.radial:
            radial_scale += power * k
            power = power * theta_sq

        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(r > 1e-12, theta * radial_scale / np.maximum(r, 1e-12), 1.0)
        xr, yr = a * scale, b * scale

        squared_norm = xr ** 2 + yr ** 2
        p0, p1 = self.tangential
        temp = 2.0 * (xr * p0 + yr * p1)
        xt = temp * xr + squared_norm * p0
        yt = temp * yr + squared_norm * p1

        s0, s1, s2, s3 = self.thin_prism
        xs = s0 * squared_norm + s1 * squared_norm ** 2
        ys = s2 * squared_norm + s3 * squared_norm ** 2

        uv = np.stack([xr + xt + xs, yr + yt + ys], axis=-1)
        pixel = self.focal * uv + self.center
        return np.where((z > 1e-9)[..., None], pixel, np.nan)


@dataclass(frozen=True)
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def project(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        z = pts[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pts[..., 0] / z + self.cx
            v = self.fy * pts[..., 1] / z + self.cy
        return np.where((z > 1e-9)[..., None], np.stack([u, v], axis=-1), np.nan)

    def as_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height}


def roll_matrix(degrees: float) -> np.ndarray:
    """광축(z) 둘레 회전. rectified 이미지를 세우는 데 쓴다."""
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rectification_map(fisheye: Fisheye624, size: int = 1024,
                      horizontal_fov_deg: float = 110.0, roll_deg: float = 0.0
                      ) -> tuple[np.ndarray, np.ndarray, PinholeCamera, np.ndarray]:
    """fisheye -> pinhole 재매핑용 (map_x, map_y), pinhole intrinsic, roll 행렬.

    출력 픽셀의 광선을 fisheye 로 **정투영**하므로 역투영 반복이 없다.
    HOT3D 는 손이 화면 아래쪽 주변부에 오는 경우가 많아 FOV 를 넉넉히 잡는다.

    Aria RGB 는 센서가 돌아가 있어 annotation 좌표계의 이미지가 옆으로 눕는다.
    WiLoR 같은 원근 모델은 똑바로 선 이미지를 가정하므로 `roll_deg` 로 세운다.
    반환된 roll 행렬 R 에 대해, annotation 카메라 좌표 p 는 rectified 좌표에서
    `R.T @ p` 이고 그 역은 `R @ p_rect` 다.
    """
    focal = (size / 2) / np.tan(np.deg2rad(horizontal_fov_deg) / 2)
    pinhole = PinholeCamera(focal, focal, size / 2, size / 2, size, size)
    rotation = roll_matrix(roll_deg)

    u, v = np.meshgrid(np.arange(size, dtype=np.float64),
                       np.arange(size, dtype=np.float64))
    rays = np.stack([(u - pinhole.cx) / pinhole.fx,
                     (v - pinhole.cy) / pinhole.fy,
                     np.ones_like(u)], axis=-1)
    source = fisheye.project(rays @ rotation.T)
    return (source[..., 0].astype(np.float32), source[..., 1].astype(np.float32),
            pinhole, rotation)


def verify_projection(fisheye: Fisheye624, joints_camera: np.ndarray,
                      boxes: np.ndarray) -> dict:
    """GT joint 투영 bbox 와 dataset 이 배포한 2D bbox 를 비교한다.

    joints_camera: (T, 21, 3), boxes: (T, 4) = [x_min, x_max, y_min, y_max]
    두 bbox 는 정의가 조금 다르다(dataset box 는 mesh 기준, 우리는 21 joint 기준)이므로
    완전 일치하지는 않지만, 카메라 모델이 틀리면 수십~수백 픽셀로 벌어진다.
    """
    pixels = fisheye.project(joints_camera)                       # (T, 21, 2)
    finite = np.isfinite(pixels).all(axis=(1, 2))
    pixels = pixels[finite]
    boxes = boxes[finite]

    predicted = np.stack([pixels[..., 0].min(1), pixels[..., 0].max(1),
                          pixels[..., 1].min(1), pixels[..., 1].max(1)], axis=1)
    center_error = np.linalg.norm(
        np.stack([(predicted[:, 0] + predicted[:, 1]) / 2 - (boxes[:, 0] + boxes[:, 1]) / 2,
                  (predicted[:, 2] + predicted[:, 3]) / 2 - (boxes[:, 2] + boxes[:, 3]) / 2],
                 axis=1), axis=1)
    inside = ((pixels[..., 0] >= boxes[:, 0:1] - 20) & (pixels[..., 0] <= boxes[:, 1:2] + 20)
              & (pixels[..., 1] >= boxes[:, 2:3] - 20) & (pixels[..., 1] <= boxes[:, 3:4] + 20))
    return {
        "n_frames": int(finite.sum()),
        "bbox_center_error_px_median": float(np.median(center_error)),
        "bbox_center_error_px_p95": float(np.percentile(center_error, 95)),
        "joints_inside_gt_box_ratio": float(inside.mean()),
    }
