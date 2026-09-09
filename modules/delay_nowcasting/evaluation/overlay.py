"""예측 자세와 GT 자세를 실제 이미지 위에 겹쳐 그린다.

  python -m modules.delay_nowcasting.evaluation.overlay \
      --config modules/delay_nowcasting/configs/data/hot3d_v1.yaml --split val

배포에서 실제로 묻는 질문은 "렌더링된 손이 진짜 손과 맞는가" 다. 그래서 예측 시점
(= target frame) 의 이미지 위에 GT 와 예측을 함께 그린다. 숫자로는 안 보이는 실패 양상
(지연이 남는지, 손가락이 흐트러지는지, 통째로 어긋나는지)이 여기서 드러난다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from ..config import REPO_ROOT, load_config
from ..data.adapters.aria_camera import Fisheye624, rectification_map
from ..data.adapters.wilor_infer import MP4_TO_ANNOTATION_ROTATION
from ..data.build_windows import gather, load_canonical_frames
from ..data.canonical_schema import BONES, NUM_JOINTS, WRIST, invert_transform, transform_points
from ..methods.baselines import BASELINES

GT_COLOR = (90, 220, 90)            # BGR, 초록 = 정답
CROP = 260


def _draw(image, pixels, color, thickness=2, radius=3):
    for a, b in BONES:
        if np.isfinite(pixels[a]).all() and np.isfinite(pixels[b]).all():
            cv2.line(image, tuple(pixels[a].astype(int)), tuple(pixels[b].astype(int)),
                     color, thickness, cv2.LINE_AA)
    for u, v in pixels:
        if np.isfinite(u):
            cv2.circle(image, (int(u), int(v)), radius, color, -1, cv2.LINE_AA)


def _predictions(history, times, horizon, methods, models, device):
    import torch

    out = {}
    for name in methods:
        out[name] = BASELINES[name](history, times, horizon)
    for name, model in models.items():
        with torch.no_grad():
            hand = torch.ones(len(history), dtype=torch.long, device=device)
            vis = torch.ones(history.shape[:3], dtype=torch.float32, device=device)
            out[name] = model(
                torch.as_tensor(history, dtype=torch.float32, device=device),
                torch.as_tensor(times, dtype=torch.float32, device=device),
                torch.as_tensor(horizon, dtype=torch.float32, device=device),
                hand, vis).double().cpu().numpy()
    return out


def build_overlay(cfg, out_dir: Path, sequence_id: str, anchor_frame: int,
                  handedness: str, conditions: dict, methods: list[str],
                  models: dict, horizons: list[float], device: str) -> np.ndarray:
    raw_dir = Path(cfg["dataset"]["raw_root"]) / sequence_id
    fisheye = Fisheye624.from_json(raw_dir / "camera_models.json")
    map_x, map_y, pinhole, _ = rectification_map(fisheye, 1024, 110.0)

    gt_frames = load_canonical_frames(out_dir / "canonical_frames_val.parquet")
    windows = pl.read_parquet(out_dir / "windows_wilor_val.parquet")
    sub = windows.filter((pl.col("sequence_id") == sequence_id)
                         & (pl.col("handedness") == handedness))
    rows = gt_frames.with_row_index("row")
    anchor_rows = rows.filter((pl.col("sequence_id") == sequence_id)
                              & (pl.col("handedness") == handedness)
                              & (pl.col("frame_idx") == anchor_frame))["row"].to_list()
    chosen = sub.filter(pl.col("anchor_row").is_in(anchor_rows))
    chosen = chosen.filter(pl.col("requested_horizon_ms").is_in(horizons))
    if not chosen.height:
        raise ValueError(f"{sequence_id} frame {anchor_frame} 에 window 가 없다")

    # target frame 이미지들을 순차 읽기로 모은다
    target_frames = {}
    needed = {int(rows["frame_idx"].to_numpy()[r]) for r in chosen["target_row"].to_list()}
    cap = cv2.VideoCapture(str(next(raw_dir.glob("*preview_rgb.mp4"))))
    k = 0
    while k <= max(needed):
        ok, frame = cap.read()
        if not ok:
            break
        if k in needed:
            target_frames[k] = cv2.remap(cv2.rotate(frame, MP4_TO_ANNOTATION_ROTATION),
                                         map_x, map_y, cv2.INTER_LINEAR)
        k += 1
    cap.release()

    names = methods + list(models)
    palette = [(60, 60, 235), (235, 160, 40), (200, 70, 220), (40, 200, 235)]
    panels = []
    for condition, history_file in conditions.items():
        history_frames = (gt_frames if history_file is None
                          else load_canonical_frames(out_dir / history_file))
        row = []
        for horizon in horizons:
            window = chosen.filter(pl.col("requested_horizon_ms") == horizon)
            batch = gather(history_frames, window, gt_frames)
            preds = _predictions(batch["history_joints"].astype(np.float64),
                                 batch["history_time_ms"], batch["horizon_ms"],
                                 methods, models, device)

            target_row = int(window["target_row"][0])
            target_frame_idx = int(rows["frame_idx"].to_numpy()[target_row])
            world_to_cam = invert_transform(
                rows["camera_to_world"].to_numpy()[target_row].reshape(4, 4).astype(np.float64))
            image = target_frames[target_frame_idx].copy()

            gt_world = batch["target_joints"][0].astype(np.float64)
            gt_px = pinhole.project(transform_points(world_to_cam, gt_world))
            _draw(image, gt_px, GT_COLOR, 3, 4)
            for i, name in enumerate(names):
                px = pinhole.project(transform_points(world_to_cam, preds[name][0]))
                _draw(image, px, palette[i % len(palette)], 2, 3)

            centre = np.nanmean(gt_px, axis=0).astype(int)
            x0 = int(np.clip(centre[0] - CROP, 0, image.shape[1] - 2 * CROP))
            y0 = int(np.clip(centre[1] - CROP, 0, image.shape[0] - 2 * CROP))
            crop = image[y0:y0 + 2 * CROP, x0:x0 + 2 * CROP]
            crop = cv2.resize(crop, (420, 420))
            cv2.putText(crop, f"{condition}  {horizon:.0f}ms", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            row.append(crop)
        panels.append(np.hstack(row))

    canvas = np.vstack(panels)
    legend = np.zeros((46 + 26 * ((len(names) + 2) // 3), canvas.shape[1], 3), np.uint8)
    cv2.putText(legend, "GT", (40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GT_COLOR, 2, cv2.LINE_AA)
    x = 120
    for i, name in enumerate(names):
        cv2.putText(legend, name, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    palette[i % len(palette)], 2, cv2.LINE_AA)
        x += 24 + 13 * len(name)
    return np.vstack([legend, canvas])


def main() -> None:
    ap = argparse.ArgumentParser(description="예측 vs GT 자세 이미지 overlay")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--figure-dir", default=None)
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--anchor-frame", type=int, default=None)
    ap.add_argument("--handedness", default="RIGHT")
    ap.add_argument("--horizons-ms", nargs="*", type=float, default=[33, 66, 100, 133])
    ap.add_argument("--methods", nargs="*", default=["hold", "cv_robust_clip"])
    ap.add_argument("--checkpoint-a", default=None, help="조건 A 용 학습 모델")
    ap.add_argument("--checkpoint-c", default=None, help="조건 C 용 학습 모델")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from ..training import checkpoint as ckpt

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    figures = Path(args.figure_dir) if args.figure_dir else REPO_ROOT / "output" / "delay_nowcasting"
    figures.mkdir(parents=True, exist_ok=True)

    windows = pl.read_parquet(out_dir / "windows_wilor_val.parquet")
    frames = load_canonical_frames(out_dir / "canonical_frames_val.parquet").with_row_index("row")
    if args.sequence is None or args.anchor_frame is None:
        # 움직임이 빠른 anchor 를 고른다. 정지 상태에서는 어떤 방법이든 똑같이 보인다.
        # 속도는 **GT** 로 잰다. WiLoR 로 재면 추정이 튄 프레임(비현실적 속도)이 뽑힌다.
        candidates = windows.filter(pl.col("requested_horizon_ms") == 100.0)
        batch = gather(frames.drop("row"), candidates, frames.drop("row"))
        from .metrics import wrist_speed_mps

        speed = wrist_speed_mps(batch["history_joints"].astype(np.float64),
                                batch["history_time_ms"])
        # 상위 1% 지점을 쓴다. 최댓값은 라벨 이상치일 수 있다.
        order = np.argsort(speed)
        pick = candidates[int(order[int(len(order) * 0.99)])]
        sequence_id = args.sequence or pick["sequence_id"][0]
        handedness = pick["handedness"][0]
        anchor_frame = int(frames["frame_idx"].to_numpy()[int(pick["anchor_row"][0])])
        print(f"자동 선택: {sequence_id} {handedness} frame {anchor_frame} "
              f"(wrist 속도 {speed.max():.2f} m/s)")
    else:
        sequence_id, anchor_frame, handedness = args.sequence, args.anchor_frame, args.handedness

    models_a = ({Path(args.checkpoint_a).parent.parent.name: ckpt.load(Path(args.checkpoint_a), args.device)[0]}
                if args.checkpoint_a else {})
    models_c = ({Path(args.checkpoint_c).parent.parent.name: ckpt.load(Path(args.checkpoint_c), args.device)[0]}
                if args.checkpoint_c else {})

    canvas_a = build_overlay(cfg, out_dir, sequence_id, anchor_frame, handedness,
                             {"A: GT history": None}, args.methods, models_a,
                             args.horizons_ms, args.device)
    canvas_c = build_overlay(cfg, out_dir, sequence_id, anchor_frame, handedness,
                             {"C: WiLoR history": "wilor_frames_val.parquet"},
                             args.methods, models_c, args.horizons_ms, args.device)
    path = figures / "forecast_overlay.png"
    cv2.imwrite(str(path), np.vstack([canvas_a, canvas_c]))
    print(f"figure : {path}")


if __name__ == "__main__":
    main()
