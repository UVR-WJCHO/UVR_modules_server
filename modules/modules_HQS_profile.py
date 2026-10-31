"""HQS 프로파일 — 한 사람의 HQS 패널 값과 모드를 시간대별 통계로 쌓는다.

평소와 다른 상태를 알아보거나 판정을 개인에 맞추는 데 쓸 기준이다. 원시 기록은 남기지 않고
크기가 고정된 통계만 둔다. 파일 하나가 한 사람이다(JSON).

  months[YYYY-MM]               현지 시각 기준 달. 이번 달과 지난 KEEP_MONTHS 개월만 두고 더 오래된 달은 지운다
    places[장소]                 그 달에 그 장소에서 관측한 초(s), 위치(loc "위도,경도"), 시간대(tz)
    cells[weekday|weekend][시]   평일/주말 x 현지 0~23 시. 관측하지 않은 칸은 null
      s                         관측한 초(HL2DATA 가 들어오던 시간)
      watch_s                   그중 와치가 붙어 있던 초
      mode                      모드(FOCUS / MOVING / TALKING / NONE ...)별 초
      emotion physical social digital bpm
                                [n, 합, 제곱합]. 값이 없던 초(와치 없음, digital 미정 등)는 세지 않는다

[n, 합, 제곱합] 은 달·칸끼리 더하기만 하면 합쳐지므로 원하는 기간·시간대의 평균과 분산을 바로 낸다.
크기는 기록 기간과 무관하게 최대 (KEEP_MONTHS + 1) x 48 칸, 수백 KB 이하다.

위치는 서버 공인 IP 로 시작할 때 한 번 조회한 도시 수준이다. HL2 가 서버와 같은 망에 있어야 하므로
서버 위치를 사용자 위치로 본다. 조회가 안 되면 장소는 "unknown", 시간대는 서버 시간대다.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
from datetime import datetime
from typing import Dict
from zoneinfo import ZoneInfo

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROFILE = os.path.join(_REPO, "output", "hqs", "profiles", "default.json")
SCHEMA = 1
KEEP_MONTHS = 6           # 이번 달 + 지난 6 개월. 가장 오래된 달도 한 달을 다 채우므로 최소 6 개월치가 남는다
VALUES = ("emotion", "physical", "social", "digital", "bpm")
HL2_STALE_S = 5.0         # HL2DATA 가 이만큼 끊기면 그동안은 관측하지 않은 것으로 본다
GEO_URL = "https://ipinfo.io/json"   # 공인 IP 의 도시·위경도·시간대. 토큰 없이 쓴다


def locate(timeout: float = 3.0) -> Dict:
    """서버 공인 IP 로 도시 수준 위치를 조회한다. {"place", "loc", "tz"}."""
    try:
        with urllib.request.urlopen(GEO_URL, timeout=timeout) as f:
            g = json.load(f)
        ZoneInfo(g["timezone"])
        place = ", ".join(g[k] for k in ("city", "region", "country") if g.get(k))
        return {"place": place or "unknown", "loc": g.get("loc"), "tz": g["timezone"]}
    except (OSError, ValueError, KeyError) as e:
        print(f"[PROFILE] 위치 조회 실패({e}). 장소 unknown, 서버 시간대로 기록한다", flush=True)
        return {"place": "unknown", "loc": None, "tz": None}


class HQSProfile:
    """add() 로 1 초 관측을 더하고 save() 로 쓴다. 파일이 이미 있으면 읽어서 그 위에 이어 쌓는다."""

    def __init__(self, path: str, where: Dict):
        self.path, self.where = path, where
        self.tz = ZoneInfo(where["tz"]) if where["tz"] else None
        self._lock = threading.Lock()
        self.is_new = not os.path.exists(path)
        if self.is_new:
            now = datetime.now(self.tz).isoformat(timespec="seconds")
            self.data = {"schema": SCHEMA, "name": os.path.splitext(os.path.basename(path))[0],
                         "created": now, "updated": now, "months": {}}
        else:                                  # 읽지 못하면 여기서 멈춘다. 새 기록으로 덮어쓰지 않는다
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)
            if self.data.get("schema") != SCHEMA:
                raise ValueError(f"{path}: schema {self.data.get('schema')} 는 읽지 못한다(기대 {SCHEMA})")

    def hours(self) -> float:
        with self._lock:
            return sum(c["s"] for m in self.data["months"].values()
                       for row in m["cells"].values() for c in row if c) / 3600.0

    def add(self, t: float, mode: str, snap: Dict) -> None:
        """t(unix s) 의 1 초 관측 하나를 더한다. snap 은 HQSPanel.snapshot()."""
        if t - snap["t_hl2"] > HL2_STALE_S:
            return
        now = datetime.fromtimestamp(t, self.tz)
        key = now.strftime("%Y-%m")
        with self._lock:
            months = self.data["months"]
            if key not in months:
                months[key] = {"places": {}, "cells": {"weekday": [None] * 24, "weekend": [None] * 24}}
                oldest = now.year * 12 + now.month - 1 - KEEP_MONTHS
                for k in [k for k in months if int(k[:4]) * 12 + int(k[5:7]) - 1 < oldest]:
                    del months[k]
            m = months[key]
            p = m["places"].setdefault(self.where["place"],
                                       {"s": 0, "loc": self.where["loc"], "tz": self.where["tz"]})
            p["s"] += 1
            row = m["cells"]["weekend" if now.weekday() >= 5 else "weekday"]
            if row[now.hour] is None:
                row[now.hour] = {"s": 0, "watch_s": 0, "mode": {}}
            c = row[now.hour]
            c["s"] += 1
            c["watch_s"] += int(snap["watch"])
            c["mode"][mode] = c["mode"].get(mode, 0) + 1
            for k in VALUES:
                v = snap[k]
                if v is not None:
                    st = c.setdefault(k, [0, 0.0, 0.0])
                    st[0] += 1
                    st[1] += v
                    st[2] += v * v

    def save(self) -> None:
        """임시 파일에 다 쓴 뒤 바꿔 넣는다. 쓰는 도중 꺼져도 앞 파일이 그대로 남는다."""
        with self._lock:
            self.data["updated"] = datetime.now(self.tz).isoformat(timespec="seconds")
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, self.path)
