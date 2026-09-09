"""2.5D 예측을 목표 프레임 이미지 위에 GT 와 함께 겹쳐 그린다.

2.5D 는 그 자체가 image 좌표라 카메라 pose 도 depth 도 필요 없다 — 초점거리만 곱하면
픽셀이 된다. 겹쳐 그리는 자세는 GT, 가장 강한 비학습 baseline, 제안 모델 셋이다.

  python -m modules.delay_nowcasting.evaluation.overlay_25d --dataset dexycb_v1

프레임은 **제안 모델이 baseline 을 가장 크게 앞서는 곳**에서 고른다. 정지 상태에서는
어떤 방법이든 같아 보이므로 그림이 아무것도 말해주지 않는다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch

from ..config import REPO_ROOT, load_config
from ..data.adapters.dexycb import read_intrinsics
from ..data.canonical_schema import BONES, NUM_JOINTS, WRIST
from ..methods.baselines import constant_velocity, kalman_constant_velocity
from ..training import checkpoint as ckpt
from ..training.dataset import load_split

GT_COLOR = (90, 220, 90)          # BGR 초록 — 표시 시점 정답
HOLD_COLOR = (60, 60, 235)        # 빨강   — 보상 없이 그대로 쓴 추정
OURS_COLOR = (235, 160, 40)       # 파랑   — 제안
DR_COLOR = (200, 70, 220)         # 자주   — dead reckoning (등속 / 칼만)


def draw(image, px, color, thickness=2, radius=3):
    for a, b in BONES:
        if np.isfinite(px[a]).all() and np.isfinite(px[b]).all():
            cv2.line(image, tuple(px[a].astype(int)), tuple(px[b].astype(int)),
                     color, thickness, cv2.LINE_AA)
    for u, v in px:
        if np.isfinite(u):
            cv2.circle(image, (int(u), int(v)), radius, color, -1, cv2.LINE_AA)


def to_pixels(pose_25d, fx, fy, cx, cy):
    """(21, 3) 2.5D -> (21, 2) 픽셀. 앞 두 채널이 곧 정규화 image 좌표다."""
    return np.stack([pose_25d[:, 0] * fx + cx, pose_25d[:, 1] * fy + cy], axis=-1)


class FrameSource:
    """데이터셋마다 이미지 저장 방식이 달라 여기서 흡수한다.

      DexYCB  : 프레임별 jpg. intrinsic 은 카메라 serial 로 읽는다.
      HOT3D   : preview mp4 + fisheye 왜곡. 주석과 같은 pinhole 로 rectify 해야
                2.5D 좌표가 맞는다.
      HOI4D   : mp4. 카메라별 intrinsic 하나로 시퀀스 전체를 쓴다.

    mp4 는 임의 접근이 어긋날 수 있어 순차 디코딩하고, 시퀀스별로 한 번만 읽어 캐시한다.
    """

    def __init__(self, cfg):
        self.name = cfg.name
        self.dataset = cfg["dataset"]
        self.root = Path(self.dataset.get("raw_root") or self.dataset["root"])
        self._video: dict[str, dict[int, np.ndarray]] = {}
        self._K: dict[str, tuple] = {}

    # -- mp4 를 통째로 읽어 프레임 dict 로 둔다 (시퀀스 하나가 수백 프레임이라 감당된다)
    def _decode(self, path: Path, rotate=None, remap=None):
        cap, out, k = cv2.VideoCapture(str(path)), {}, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if rotate is not None:
                frame = cv2.rotate(frame, rotate)
            if remap is not None:
                frame = cv2.remap(frame, remap[0], remap[1], cv2.INTER_LINEAR)
            out[k] = frame
            k += 1
        cap.release()
        return out

    def get(self, sequence_id: str, frame_idx: int):
        """(BGR image, (fx, fy, cx, cy)) 또는 (None, None)."""
        if self.name.startswith("dexycb"):
            image, serial = dexycb_image(self.root, sequence_id, frame_idx)
            if image is None:
                return None, None
            i = read_intrinsics(Path(self.dataset["calibration_root"]), serial)
            return image, (i["fx"], i["fy"], i["cx"], i["cy"])

        if self.name.startswith("hot3d"):
            from ..data.adapters.aria_camera import Fisheye624, rectification_map
            from ..data.adapters.wilor_infer import MP4_TO_ANNOTATION_ROTATION

            if sequence_id not in self._video:
                raw = self.root / sequence_id
                fisheye = Fisheye624.from_json(raw / "camera_models.json")
                map_x, map_y, pinhole, _ = rectification_map(fisheye, 512, 110.0)
                self._K[sequence_id] = (pinhole.fx, pinhole.fy, pinhole.cx, pinhole.cy)
                self._video[sequence_id] = self._decode(
                    next(raw.glob("*preview_rgb.mp4")),
                    rotate=MP4_TO_ANNOTATION_ROTATION, remap=(map_x, map_y))
            return self._video[sequence_id].get(frame_idx), self._K[sequence_id]

        if self.name.startswith("hoi4d"):
            from ..data.adapters.hoi4d import read_intrinsics as hoi4d_intrinsics

            if sequence_id not in self._video:
                camera = sequence_id.split("/")[0]
                K = hoi4d_intrinsics(self.root, camera)
                self._K[sequence_id] = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
                self._video[sequence_id] = self._decode(
                    self.root / "HOI4D_release" / sequence_id / "align_rgb" / "image.mp4")
            return self._video[sequence_id].get(frame_idx), self._K[sequence_id]

        raise SystemExit(f"이미지 로더가 없는 데이터셋: {self.name}")


def dexycb_image(raw_root: Path, sequence_id: str, frame_idx: int):
    subject_sequence, serial = sequence_id.split("@")
    subject, sequence = subject_sequence.split("/")
    path = raw_root / subject / sequence / serial / f"color_{frame_idx:06d}.jpg"
    return cv2.imread(str(path)), serial


def main() -> None:
    ap = argparse.ArgumentParser(description="2.5D 예측 overlay")
    ap.add_argument("--dataset", default="dexycb_v1")
    ap.add_argument("--model", default=None, help="기본은 <dataset 이름>_25d")
    ap.add_argument("--horizons-ms", nargs="*", type=float, default=[66, 133, 199])
    ap.add_argument("--columns", type=int, default=3, help="고를 프레임 수")
    ap.add_argument("--crop", type=int, default=200)
    ap.add_argument("--out", default="research/paper/figures/overlay_25d.png")
    ap.add_argument("--separate", action="store_true",
                    help="프레임마다 한 장씩 저장한다")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / f"modules/delay_nowcasting/configs/data/{args.dataset}.yaml")
    root = REPO_ROOT / "research_data" / args.dataset
    model_dir = root / (args.model or f"{args.dataset.split('_')[0]}_25d")
    source = FrameSource(cfg)

    d = load_split(root, "val", "cuda", history_file="wilor25d_frames_val.parquet",
                   target_file="canonical25d_frames_val.parquet",
                   windows_file="windows_wilor_val.parquet")
    far = float(max(args.horizons_ms))
    index = torch.as_tensor(np.flatnonzero(d.requested_horizon_ms == far), device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    b = d.batch(index, None, g)
    hist = b["history"].cpu().numpy().astype(np.float64)
    tms = b["history_time_ms"].cpu().numpy().astype(np.float64)
    tgt = b["target"].cpu().numpy().astype(np.float64)
    H = b["horizon_ms"].cpu().numpy()

    model = ckpt.load(sorted(model_dir.glob("seed*/best_model.pt"))[0], "cuda")[0].eval()
    with torch.no_grad():
        ours = model(b["history"], b["history_time_ms"], b["horizon_ms"],
                     b["handedness"], b["visibility"]).double().cpu().numpy()
    # 비교군은 보상하지 않았을 때 화면에 나오는 것, 즉 anchor 를 그대로 둔 자세다.
    # 우리가 고른 외삽 알고리즘을 세우면 비교가 임의적이 된다.
    base = hist[:, -1]

    # 우리가 앞서는 프레임을 고르되 극단은 뺀다. 대표성이 없다.
    e_base = np.linalg.norm(base[:, :, :2] - tgt[:, :, :2], axis=-1).mean(-1)
    e_ours = np.linalg.norm(ours[:, :, :2] - tgt[:, :, :2], axis=-1).mean(-1)
    hand = np.linalg.norm(tgt[:, 9, :2] - tgt[:, WRIST, :2], axis=-1)   # 손 크기(정규화)
    # 손이 가려진 프레임은 그림으로 읽히지 않는다. 추정기가 anchor 를 잘 맞춘 프레임만
    # 쓴다 — WiLoR 가 제대로 잡았다는 것은 그 시점에 손이 보였다는 뜻이다.
    gt25 = pl.read_parquet(root / "canonical25d_frames_val.parquet", columns=["joints_world"])
    gt_all = gt25["joints_world"].to_numpy().reshape(-1, NUM_JOINTS, 3).astype(np.float64)
    anchor_row = d.history_rows[index][:, -1].cpu().numpy()
    anchor_gt = gt_all[anchor_row]
    visible = np.linalg.norm(hist[:, -1, :, :2] - anchor_gt[:, :, :2], axis=-1).mean(-1)
    ok = (np.isfinite(e_base) & np.isfinite(e_ours) & np.isfinite(visible)
          & (visible < 0.20 * hand)                          # anchor 가 손 크기의 20% 이내
          & (hand > np.nanpercentile(hand, 70))              # 손이 크게 잡힌 프레임
          & (e_base < 1.5 * hand) & (e_base > 0.4 * hand))   # 손 크기의 0.4~1.5 배로 제한
    order = np.argsort(-np.where(ok, e_base - e_ours, -np.inf))

    frames = pl.read_parquet(root / "canonical_frames_val.parquet",
                             columns=["sequence_id", "frame_idx"])
    seq = frames["sequence_id"].to_numpy()
    fidx = frames["frame_idx"].to_numpy()
    target_row = d.target_row[index].cpu().numpy()

    # 아래 줄에 놓을 dead reckoning. 근거가 분명한 둘만 쓴다 — 현재 XR 런타임이
    # 표시 시각까지 외삽할 때 쓰는 방식이다 (등속 / 칼만).
    # 표(tab:main)와 같은 참조를 쓴다. n_fit 이 다르면 그림과 표가 서로 다른 baseline 이 된다.
    dr = {"Constant velocity": constant_velocity(hist, tms, H, n_fit=4),
          "Kalman (CV)": kalman_constant_velocity(hist, tms, H)}

    picked, second, used_sequences = [], [], set()
    for k in order:
        row = target_row[k]
        if seq[row] in used_sequences:          # 같은 시퀀스에서 여러 장 뽑지 않는다
            continue
        image, K = source.get(seq[row], int(fidx[row]))
        a_row = anchor_row[k]
        anchor_image, _ = source.get(seq[a_row], int(fidx[a_row]))
        if image is None or anchor_image is None:
            continue
        used_sequences.add(seq[row])
        image, anchor_image = image.copy(), anchor_image.copy()
        gt_px = to_pixels(tgt[k], *K)
        draw(anchor_image, to_pixels(base[k], *K), HOLD_COLOR, 3, 4)
        draw(image, gt_px, GT_COLOR, 3, 4)
        draw(image, to_pixels(base[k], *K), HOLD_COLOR, 2, 3)
        draw(image, to_pixels(ours[k], *K), OURS_COLOR, 2, 3)
        # crop 은 GT 손 크기로 잡는다. 예측이 그보다 멀리 나가면 잘리는데, 그 자체가
        # 손을 벗어났다는 표시라 오히려 읽기 쉽다. 예측까지 담으면 손이 너무 작아진다.
        centre = np.nanmean(gt_px, axis=0).astype(int)
        span = max(np.nanmax(np.abs(gt_px - centre)), 30)
        c = int(min(span * 2.2, min(image.shape[:2]) // 2))
        x0 = int(np.clip(centre[0] - c, 0, image.shape[1] - 2 * c))
        y0 = int(np.clip(centre[1] - c, 0, image.shape[0] - 2 * c))
        def cut(img, label):
            out = cv2.resize(img[y0:y0 + 2 * c, x0:x0 + 2 * c], (360, 360))
            cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            return out
        pair = np.hstack([cut(anchor_image, "Capture"),
                          cut(image, f"Display (with Ours)  +{int(far)} ms")])
        picked.append(cv2.copyMakeBorder(pair, 0, 0, 0, 12, cv2.BORDER_CONSTANT, value=0))
        # 같은 프레임에 dead reckoning 을 하나씩 따로 그린다. 한 장에 다 겹치면 안 읽힌다.
        cells = []
        for name, pred in dr.items():
            canvas_dr = source.get(seq[row], int(fidx[row]))[0].copy()
            draw(canvas_dr, gt_px, GT_COLOR, 3, 4)
            draw(canvas_dr, to_pixels(pred[k], *K), DR_COLOR, 2, 3)
            cells.append(cut(canvas_dr, name))
        second.append(cv2.copyMakeBorder(np.hstack(cells), 0, 0, 0, 12,
                                         cv2.BORDER_CONSTANT, value=0))
        print(f"  {seq[row]} frame {fidx[row]}  hold {e_base[k]*1000:.1f} "
              f"ours {e_ours[k]*1000:.1f} (x1e3)")
        if len(picked) == args.columns:
            break

    # 범례는 굽지 않는다. 패널마다 보이는 색이 다르고 (capture 에는 GT 가 없다) 굽은
    # 글씨는 본문 서체와 맞지 않는다. 색 설명은 논문 캡션이 맡는다.
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.separate:
        # 프레임마다 한 장씩 낸다. 논문에서 배치를 따로 잡을 때 쓴다.
        for i, (top, bottom) in enumerate(zip(picked, second), 1):
            path = out.with_name(f"{out.stem}_{i}{out.suffix}")
            cv2.imwrite(str(path), np.vstack([top, bottom]))
            print(f"저장: {path}")
    else:
        cv2.imwrite(str(out), np.vstack([np.hstack(picked), np.hstack(second)]))
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
