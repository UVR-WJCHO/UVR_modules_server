"""모든 방법이 공유하는 result 표 schema.

B 계획 §9 는 sequence 단위로 먼저 metric 을 계산하고, 같은 sequence 에서 두 방법을
paired 로 비교하며, horizon/subject/handedness/motion tercile 등 여러 축으로 나눠
보고할 것을 요구한다. 축이 계속 늘어나므로 wide 표가 아니라 long 표 하나로 둔다.
groupby 축이 추가돼도 schema 를 바꾸지 않는다.
"""
from __future__ import annotations

# history 품질 조건 (B 계획 §10.3). 결과 표는 이 세 종류를 절대 섞지 않는다.
HISTORY_SOURCES: tuple[str, ...] = ("gt", "corrupted_gt", "wilor")


def metric_result_schema() -> dict:
    import polars as pl

    return {
        # 무엇을 재현하면 이 행이 나오는가
        "method": pl.Utf8,             # hold / cv_robust / residual_mlp / tcn_lite ...
        "model_version": pl.Utf8,      # checkpoint 또는 baseline 설정 식별자
        "config_hash": pl.Utf8,
        "seed": pl.Int64,              # baseline 은 null
        # 어떤 데이터에서 쟀는가
        "split": pl.Utf8,              # train / val / test
        "history_source": pl.Utf8,     # HISTORY_SOURCES
        "sequence_id": pl.Utf8,        # sequence 단위 집계가 통계의 기본 단위 (§9.5)
        "subject_id": pl.Utf8,
        "handedness": pl.Utf8,
        "horizon_ms": pl.Float64,
        "interpolated_target": pl.Boolean,   # §4.4-7
        # 추가 groupby 축 (wrist speed tercile, head motion, occlusion ...) 은
        # schema 변경 없이 이 두 column 으로 표현한다 (§9.4)
        "group_key": pl.Utf8,
        "group_value": pl.Utf8,
        # 값
        "metric": pl.Utf8,             # abs_mpjpe_mm / root_rel_mpjpe_mm / wrist_err_mm ...
        "value": pl.Float64,
        "n_frames": pl.Int64,
    }


def validate_metric_frame(df) -> None:
    expected = metric_result_schema()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"metric 표에 없는 column: {missing}")
    for col, dtype in expected.items():
        if df.schema[col] != dtype:
            raise ValueError(f"column {col} dtype 이 {dtype} 가 아니라 {df.schema[col]} 다")

    bad = set(df["history_source"].unique().to_list()) - set(HISTORY_SOURCES)
    if bad:
        raise ValueError(f"history_source 값이 {HISTORY_SOURCES} 밖에 있다: {sorted(bad)}")
