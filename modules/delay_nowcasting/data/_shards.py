"""긴 캐시 빌드를 구간 단위로 저장해 중단·재시작을 견디게 한다.

HOI4D 는 시퀀스가 수천 개라 한 번에 두 시간 넘게 돈다. 중간에 멈추면 표를 끝에서
한 번에 쓰는 구조에서는 전부 날아간다. 그래서 구간마다 parquet 을 쓰고, 다시 시작하면
이미 있는 구간은 건너뛴다.

행 순서는 반드시 보존해야 한다 — canonical 표와 WiLoR 표를 row index 로 맞대기 때문이다.
구간을 시퀀스 목록의 고정 크기 조각으로 나누고 조각 번호 순서로 이어 붙이면 순서가
그대로 유지된다.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

CHUNK = 50


def chunks(sequences: list, size: int = CHUNK) -> list[list]:
    return [sequences[i:i + size] for i in range(0, len(sequences), size)]


def done(parts_dir: Path, index: int, expected_ids: list[str]) -> bool:
    """이 구간이 이미 저장돼 있고 시퀀스 목록도 그대로인가.

    목록이 달라졌다면(데이터가 바뀌었거나 split 을 고쳤다면) 낡은 조각이므로 다시 만든다.
    """
    meta = parts_dir / f"{index:04d}.json"
    table = parts_dir / f"{index:04d}.parquet"
    if not (meta.exists() and table.exists()):
        return False
    return json.loads(meta.read_text()).get("sequence_ids") == expected_ids


def save(parts_dir: Path, index: int, table: pl.DataFrame, infos: list[dict],
         sequence_ids: list[str]) -> None:
    parts_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(parts_dir / f"{index:04d}.parquet")
    (parts_dir / f"{index:04d}.json").write_text(
        json.dumps({"sequence_ids": sequence_ids, "sequences": infos},
                   indent=2, ensure_ascii=False))


def merge(parts_dir: Path, n_chunks: int) -> tuple[pl.DataFrame, list[dict]]:
    """조각 번호 순으로 이어 붙인다. 순서가 곧 행 대응이라 정렬을 바꾸면 안 된다."""
    tables, infos = [], []
    for index in range(n_chunks):
        tables.append(pl.read_parquet(parts_dir / f"{index:04d}.parquet"))
        infos += json.loads((parts_dir / f"{index:04d}.json").read_text())["sequences"]
    return pl.concat(tables), infos
