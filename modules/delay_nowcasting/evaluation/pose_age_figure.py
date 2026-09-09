"""실기기 세션의 렌더 시점 pose age 분포. 논문 Fig. pose-age.

  python -m modules.delay_nowcasting.evaluation.pose_age_figure

고정 horizon 하나로는 감당이 안 된다는 것을 보이는 그림이라, grid 점과 이전 grid 상한을
함께 표시한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# IEEE PDF eXpress 는 Type 3 font 를 경고한다. 42 = TrueType 으로 내보낸다.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

from ..config import REPO_ROOT

GRID = (0, 33, 66, 100, 133, 166, 199, 233, 266, 300)


def render_ages(session: Path) -> np.ndarray:
    """latency CSV 의 pose_age_ms_mono. 단조 시계이고 렌더가 끝난 뒤에 찍힌다.

    render.jsonl 의 poseAgeMs 는 Update 에서 계산돼 카메라 렌더링(26.5 ms)이 빠져 있어
    26 ms 낙관적이다. 지연 분석은 t_*_ns 계열로 통일한다.
    """
    import csv
    path = next(session.glob("latency_*.csv"))
    rows = csv.DictReader(path.open(encoding="utf-8-sig"))
    return np.array([float(r["pose_age_ms_mono"]) for r in rows
                     if str(r.get("pose_age_ms_mono", "")).strip()])


def main() -> None:
    ap = argparse.ArgumentParser(description="pose age 분포 그림")
    # 논문이 보고하는 네 세션. tab:latency 와 같은 표본이어야 한다.
    ap.add_argument("--sessions", nargs="+", default=[
        "research/eval/sess_20260828_084112", "research/eval/sess_20260828_085106",
        "research/eval/sess_20260830_081029", "research/eval/sess_20260830_081446"])
    ap.add_argument("--out", default="research/paper/figures/pose_age.pdf")
    args = ap.parse_args()

    ages = np.concatenate([render_ages(REPO_ROOT / s) for s in args.sessions])
    lo, hi = 60, 400
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.hist(ages, bins=np.arange(lo, hi + 6, 6), color="0.55", edgecolor="none")
    for x in GRID[1:]:
        ax.axvline(x, color="0.85", lw=0.5, zorder=0)
    ax.axvline(300, color="tab:red", lw=1.2,
               label=f"grid limit 300 ms ({np.mean(ages > 300) * 100:.1f}% beyond)")
    ax.axvline(np.median(ages), color="k", lw=1.0, ls=":",
               label=f"median {np.median(ages):.0f} ms")
    ax.set_xlabel("Pose age at render (ms)", fontsize=7)
    ax.set_ylabel("Frames", fontsize=7)
    ax.set_xlim(lo, hi)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.legend(fontsize=6.5, frameon=True, framealpha=0.9, edgecolor="none",
              facecolor="white", loc="upper right", borderpad=0.3,
              handlelength=1.4, handletextpad=0.5, labelspacing=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"{len(ages)} 프레임  p50 {np.median(ages):.1f}  p90 {np.percentile(ages,90):.1f}  "
          f"p99 {np.percentile(ages,99):.1f}  >199ms {np.mean(ages>199)*100:.0f}%  "
          f">300ms {np.mean(ages>300)*100:.1f}%")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
