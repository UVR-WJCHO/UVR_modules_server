"""여러 도메인의 frame·window 표를 하나로 이어 붙인다.

  python -m modules.delay_nowcasting.data.build_mixed \
      --out mixed3_v1 --source dexycb_v1 hot3d_v1 hoi4d_v1 --splits train val

window 는 frame 표의 **행 번호**로 history 와 target 을 가리킨다. 그래서 표를 이어 붙일
때 뒤 도메인의 행 번호를 앞 도메인들의 행 수만큼 밀어야 한다. 이것을 빠뜨리면 엉뚱한
프레임을 target 으로 읽는데, 오류 없이 조용히 틀린 값이 나온다.

sequence_id 규약이 도메인마다 달라(HOT3D "P0012_...", DexYCB "2020...@serial",
HOI4D "ZY2021.../H*/C*/...") track 이 서로 충돌하지 않는다.
"""
from __future__ import annotations

import argparse

import polars as pl

from ..config import REPO_ROOT

FRAME_FILES = ("wilor_frames", "canonical_frames", "wilor25d_frames", "canonical25d_frames")
# frame 표의 행 번호를 담는 컬럼은 전부 여기 있어야 한다. 하나라도 빠지면 그 컬럼만
# 앞 도메인을 가리켜 조용히 엉뚱한 frame 을 읽는다.
ROW_COLUMNS = ("anchor_row", "target_row", "target_row2")


def main() -> None:
    ap = argparse.ArgumentParser(description="도메인 여러 개를 하나의 데이터셋으로 합친다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", nargs="+", required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--windows", default="windows_wilor")
    args = ap.parse_args()

    root = REPO_ROOT / "research_data"
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        offset, frames, windows = 0, {n: [] for n in FRAME_FILES}, []
        for source in args.source:
            counts = set()
            for name in FRAME_FILES:
                path = root / source / f"{name}_{split}.parquet"
                table = pl.read_parquet(path)
                counts.add(table.height)
                frames[name].append(table)
            if len(counts) != 1:
                raise SystemExit(f"{source}/{split}: frame 표 행 수가 서로 다르다 {counts}")
            n_rows = counts.pop()

            w = pl.read_parquet(root / source / f"{args.windows}_{split}.parquet")
            # 보간 컬럼은 15 fps 도메인에만 있다. offset **전에** 채워야 함께 밀린다.
            if "target_row2" not in w.columns:
                w = w.with_columns([pl.col("target_row").alias("target_row2"),
                                    pl.lit(0.0, dtype=pl.Float32).alias("target_weight")])
            w = w.with_columns(
                [pl.col(c) + offset for c in ROW_COLUMNS]
                + [pl.col("history_rows").list.eval(pl.element() + offset)
                   if w.schema["history_rows"] == pl.List else
                   pl.col("history_rows").arr.to_list().list.eval(pl.element() + offset)
                     .list.to_array(w.schema["history_rows"].size)])
            windows.append(w.select(sorted(w.columns)))
            print(f"  [{split}] {source}: frame {n_rows:,}  window {w.height:,}  "
                  f"offset {offset:,}", flush=True)
            offset += n_rows

        for name, parts in frames.items():
            pl.concat(parts).write_parquet(out_dir / f"{name}_{split}.parquet")
        combined = pl.concat(windows)
        combined.write_parquet(out_dir / f"{args.windows}_{split}.parquet")
        print(f"  [{split}] 합계 frame {offset:,}  window {combined.height:,} -> {out_dir}")


if __name__ == "__main__":
    main()
