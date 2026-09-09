"""Phase 1 시각화 (B 계획 §11 Phase 1, §14).

  python -m modules.delay_nowcasting.evaluation.figures \
      --config modules/delay_nowcasting/configs/data/hot3d_v1.yaml --split val

생성물:
  figures/skeleton_check.png   canonical skeleton + UmeTrack landmark 겹쳐보기 (joint 매핑 확인)
  figures/palm_frame_check.png palm frame 에서 좌/우 articulation 구조 대조
  figures/error_by_horizon.png baseline horizon-error curve
  figures/error_by_motion.png  wrist speed tercile 별 horizon-error curve
  figures/sequence_examples.png 한 sequence 의 wrist 궤적과 예측
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..config import REPO_ROOT, load_config
from ..data.adapters.hot3d import UMETRACK_CHECK_PAIRS
from ..data.build_windows import gather, load_canonical_frames
from ..data.canonical_schema import BONES, JOINT_NAMES, NUM_JOINTS
from ..methods.baselines import BASELINES
from .evaluate import summarize

FINGER_COLORS = ("#d1495b", "#edae49", "#66a182", "#2e4057", "#8d6a9f")


def _draw_hand(ax, joints, landmarks=None, title=""):
    for finger, color in enumerate(FINGER_COLORS):
        chain = [0] + [1 + finger * 4 + k for k in range(4)]
        pts = joints[chain]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color=color, markersize=3, linewidth=1.5)
    ax.scatter(*joints[0], color="black", s=40, label="wrist")
    if landmarks is not None:
        ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2],
                   marker="x", color="gray", s=25, label="UmeTrack")
    for idx in (1, 4, 5, 20):
        ax.text(*joints[idx], f" {idx}:{JOINT_NAMES[idx]}", fontsize=6)
    ax.set_title(title, fontsize=9)
    ax.set_box_aspect((1, 1, 1))
    span = np.ptp(joints, axis=0).max() * 0.6
    center = joints.mean(axis=0)
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - span, c + span)
    ax.tick_params(labelsize=5)


def skeleton_check(cfg, frames: pl.DataFrame, out_path: Path, n_frames: int = 3) -> None:
    """canonical skeleton 과 UmeTrack landmark 를 겹쳐 좌/우와 joint 매핑을 눈으로 확인."""
    dataset = cfg["dataset"]
    sequence_id = frames["sequence_id"][0]
    cache_dir = (Path(dataset["cache_root"]) / "sequences"
                 / f"{sequence_id}__dev{dataset['device_index']}")
    cache_hands = pl.read_parquet(cache_dir / "hands.parquet")

    fig = plt.figure(figsize=(4 * n_frames, 8))
    for row, handedness in enumerate(("LEFT", "RIGHT")):
        sub = frames.filter((pl.col("sequence_id") == sequence_id)
                            & (pl.col("handedness") == handedness) & pl.col("valid"))
        picks = [int(i) for i in np.linspace(0, sub.height - 1, n_frames)]
        prefix = "L" if handedness == "LEFT" else "R"
        for col, pick in enumerate(picks):
            joints = sub["joints_world"].to_numpy()[pick].reshape(NUM_JOINTS, 3)
            ts = sub["timestamp_ns"][pick]
            marks = cache_hands.filter(pl.col("timestamp_ns") == ts)
            landmarks = np.stack([
                marks.filter(pl.col("joint_label") == f"{prefix}_{label}")
                     .select(["x", "y", "z"]).to_numpy()[0]
                for label, _ in UMETRACK_CHECK_PAIRS])
            ax = fig.add_subplot(2, n_frames, row * n_frames + col + 1, projection="3d")
            _draw_hand(ax, joints, landmarks, f"{handedness}  frame {pick}")
    fig.suptitle(f"{sequence_id}: canonical MANO skeleton vs UmeTrack landmark", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def palm_frame(joints: np.ndarray) -> np.ndarray:
    """(..., 21, 3) world -> palm 기준 local 좌표 (B 계획 §3.3)."""
    wrist = joints[..., 0, :]
    x = joints[..., 5, :] - joints[..., 17, :]              # index_mcp - little_mcp
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y0 = joints[..., 9, :] - wrist                          # middle_mcp - wrist
    z = np.cross(x, y0)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z], axis=-2)                 # (..., 3, 3) rows = axes
    return np.einsum("...ij,...nj->...ni", rotation, joints - wrist[..., None, :])


def palm_frame_check(frames: pl.DataFrame, out_path: Path, n_frames: int = 40) -> None:
    """palm 기준 좌표에서 좌/우 articulation 구조를 비교한다.

    palm frame 은 x 축을 `index_mcp - little_mcp` 로 잡으므로 chirality 를 정의상
    제거한다. 따라서 좌/우는 거울상이 아니라 **겹쳐야** 정상이고, 어긋나면 한쪽의
    joint 순서나 mirroring 이 깨진 것이다. 실제 좌우 거울 관계는 world 좌표의
    `canonical_schema.chirality` 부호로 검사한다.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    local = {}
    for handedness in ("LEFT", "RIGHT"):
        sub = frames.filter((pl.col("handedness") == handedness) & pl.col("valid"))
        picks = [int(i) for i in np.linspace(0, sub.height - 1, n_frames)]
        joints = sub["joints_world"].to_numpy()[picks].reshape(-1, NUM_JOINTS, 3).astype(np.float64)
        local[handedness] = palm_frame(joints)

    for ax, (handedness, color) in zip(axes[:2], (("LEFT", "#2e4057"), ("RIGHT", "#d1495b"))):
        pts = local[handedness]
        for finger in range(5):
            chain = [0] + [1 + finger * 4 + k for k in range(4)]
            ax.plot(pts[:, chain, 0].T, pts[:, chain, 1].T, color=color, alpha=0.15, linewidth=1)
        ax.plot(pts[:, [0, 1, 2, 3, 4], 0].mean(0), pts[:, [0, 1, 2, 3, 4], 1].mean(0),
                "-o", color="black", linewidth=2, markersize=4, label="thumb (mean)")
        ax.set_title(f"{handedness}: palm frame, {n_frames} frames", fontsize=10)
        ax.legend(fontsize=8)

    for handedness, color in (("LEFT", "#2e4057"), ("RIGHT", "#d1495b")):
        mean = local[handedness].mean(axis=0)
        for finger in range(5):
            chain = [0] + [1 + finger * 4 + k for k in range(4)]
            axes[2].plot(mean[chain, 0], mean[chain, 1], "-o", color=color,
                         markersize=4, linewidth=1.5,
                         label=handedness if finger == 0 else None)
    axes[2].set_title("LEFT vs RIGHT mean pose in own palm frame (must coincide)", fontsize=10)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("palm x: little_mcp -> index_mcp (m)")
        ax.set_ylabel("palm y: wrist -> middle_mcp (m)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _curve(ax, table, group_key, group_value, title):
    summary = summarize(table, group_key=group_key, group_value=group_value)
    horizons = sorted(float(c) for c in summary.columns if c != "method")
    for method in summary["method"]:
        row = summary.filter(pl.col("method") == method)
        values = [row[str(h) if str(h) in summary.columns else f"{h}"][0] for h in horizons]
        ax.plot(horizons, values, "-o", markersize=4, label=method)
    ax.set_xlabel("horizon (ms)")
    ax.set_ylabel("absolute MPJPE (mm)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def error_by_horizon(table: pl.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _curve(ax, table, "all", "all", "Baseline horizon-error curve (GT history)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def error_by_motion(table: pl.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, tercile in zip(axes, ("low", "mid", "high")):
        _curve(ax, table, "wrist_speed_tercile", tercile, f"wrist speed: {tercile}")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def sequence_examples(frames: pl.DataFrame, windows: pl.DataFrame, out_path: Path,
                      horizon_ms: float = 100.0, n_samples: int = 400) -> None:
    """한 track 에서 wrist 궤적과 각 baseline 의 예측을 시간축으로 겹쳐 본다."""
    sequence_id = frames["sequence_id"][0]
    sub = windows.filter((pl.col("sequence_id") == sequence_id)
                         & (pl.col("handedness") == "RIGHT")
                         & (pl.col("requested_horizon_ms") == horizon_ms))
    sub = sub[: n_samples]
    batch = gather(frames, sub)
    history = batch["history_joints"].astype(np.float64)
    times = batch["history_time_ms"]
    horizon = batch["horizon_ms"]
    target = batch["target_joints"]

    t = np.arange(sub.height)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, target[:, 0, 0], color="black", linewidth=2, label="GT wrist x")
    axes[0].plot(t, history[:, -1, 0, 0], color="gray", linestyle="--", linewidth=1,
                 label="anchor wrist x (= Hold)")
    for name in ("cv_2frame", "cv_robust", "const_accel", "kalman_cv"):
        axes[0].plot(t, BASELINES[name](history, times, horizon)[:, 0, 0],
                     linewidth=1, alpha=0.8, label=name)
    axes[0].set_ylabel("wrist x (m)")
    axes[0].set_title(f"{sequence_id} RIGHT, horizon {horizon_ms:.0f} ms", fontsize=10)
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(alpha=0.3)

    for name in ("hold", "cv_2frame", "const_accel", "kalman_cv"):
        error = np.linalg.norm(BASELINES[name](history, times, horizon) - target,
                               axis=-1).mean(axis=-1) * 1000
        axes[1].plot(t, error, linewidth=1, alpha=0.85, label=name)
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("absolute MPJPE (mm)")
    axes[1].legend(fontsize=8, ncol=4)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def paired_plot(table: pl.DataFrame, method: str, out_path: Path,
                horizons_ms=(33.0, 66.0, 100.0, 133.0),
                metric: str = "abs_mpjpe_mm") -> None:
    """sequence-hand 단위 paired 비교 (B 계획 §11 Phase 2 산출물, §9.5).

    점 하나가 (sequence, hand) 하나다. 대각선 아래면 학습 모델이 그 sequence 에서
    baseline 을 이겼다는 뜻이다.
    """
    from .bootstrap import best_reference, paired_units

    fig, axes = plt.subplots(1, len(horizons_ms), figsize=(4.2 * len(horizons_ms), 4.4))
    for ax, horizon in zip(np.atleast_1d(axes), horizons_ms):
        reference = best_reference(table, list(BASELINES), metric, horizon)
        units = paired_units(table, method, reference, metric, horizon)
        x, y = units[reference].to_numpy(), units[method].to_numpy()
        limit = max(x.max(), y.max()) * 1.08

        ax.plot([0, limit], [0, limit], color="gray", linewidth=1)
        ax.scatter(x, y, s=22, alpha=0.75, color="#2e4057", edgecolor="white", linewidth=0.5)
        wins = int((y < x).sum())
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{reference} (mm)")
        ax.set_ylabel(f"{method} (mm)")
        ax.set_title(f"{horizon:.0f} ms - model wins {wins}/{len(x)}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{metric}, GT history (condition A). Below diagonal = learned model better",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 figures")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", default=None, help="canonical cache/window/metric 이 있는 곳")
    ap.add_argument("--figure-dir", default=None,
                    help="그림 저장 위치 (기본: 저장소 최상위 output/delay_nowcasting)")
    ap.add_argument("--tag", default="baseline", help="어느 metric 표를 쓸지")
    ap.add_argument("--paired-method", default=None,
                    help="지정하면 이 method 의 sequence 단위 paired plot 을 추가로 그린다")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg["output_root"] / cfg.name
    # 그림은 research/ 안쪽 깊은 outputs/ 대신 눈에 띄는 저장소 최상위에 둔다.
    figures = Path(args.figure_dir) if args.figure_dir else REPO_ROOT / "output" / "delay_nowcasting"
    figures.mkdir(parents=True, exist_ok=True)

    frames = load_canonical_frames(out_dir / f"canonical_frames_{args.split}.parquet")
    windows = pl.read_parquet(out_dir / f"windows_{args.split}.parquet")
    table = pl.read_parquet(out_dir / f"{args.tag}_metrics_{args.split}.parquet")

    skeleton_check(cfg, frames, figures / "skeleton_check.png")
    palm_frame_check(frames, figures / "palm_frame_check.png")
    error_by_horizon(table, figures / "error_by_horizon.png")
    error_by_motion(table, figures / "error_by_motion.png")
    sequence_examples(frames, windows, figures / "sequence_examples.png")
    if args.paired_method:
        paired_plot(table, args.paired_method, figures / "paired_vs_baseline.png")
    for path in sorted(figures.glob("*.png")):
        print(f"figure : {path}")


if __name__ == "__main__":
    main()
