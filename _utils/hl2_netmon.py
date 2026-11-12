"""HL2 네트워크 연결 모니터. HL2 를 들고 이동하며 어디서 comm_hub 와 끊기는지 본다.

comm_hub 에서 HL2DATA 를 구독해 1 초마다 한 줄을 남긴다.
  받은 프레임 수, kB/s, 최대 도착 간격, 상대 지연, HL2 머리 위치, 출발점부터 수평 거리
HL2 는 네트워크가 끊겨도 자기 위치를 추적하므로 끊긴 곳을 위치로 남길 수 있다.
출발점은 처음 받은 프레임의 위치다. 서버·공유기 가까이에서 HL2 앱을 켜고 출발한다.

  GAP_S 넘게 프레임이 없으면 "끊김"(마지막 프레임의 위치·거리), 다시 오면 "복구"(끊긴 시간·위치)를 남긴다.
  상대 지연 = (도착 시각 - HL2 timestamp) 에서 지금까지 그 값의 최솟값을 뺀 것. HL2 가 끊긴 동안
  쌓아 둔 프레임을 다시 연결된 뒤 한꺼번에 보내면 도착 간격은 짧아 보이고 이 값이 튄다.

실행 (comm_hub 가 떠 있어야 한다)
    python _utils/hl2_netmon.py
로그는 콘솔과 output/netmon/<시작 시각>.log 에 같이 쓴다. Ctrl+C 로 끝내면 끊김 요약을 남긴다.
"""
import argparse
import math
import os
import sys
import time
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "_comm"))
import hl2_data_pb2 as proto                                          # noqa: E402
from hub_client import HubClient, add_broker_args, KW_HL2DATA        # noqa: E402

GAP_S = 2.0               # 이만큼 프레임이 없으면 끊긴 것으로 본다
IDENTITY = b"HL2_NETMON"


class Log:
    """시각을 붙여 콘솔과 파일에 같이 쓴다."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"{datetime.now():%H:%M:%S}  {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def fmt(p):
    return f"({p[0]:+.1f}, {p[1]:+.1f}, {p[2]:+.1f})"


def run(args):
    path = os.path.join(_ROOT, "output", "netmon", f"{datetime.now():%Y%m%d_%H%M%S}.log")
    log = Log(path)
    hub = HubClient(args.host, args.port, recv_kw=KW_HL2DATA, identity=IDENTITY, queue_size=1000)
    log(f"로그 {path}. HL2DATA 기다리는 중")

    pkt = proto.HL2SensorPacket()
    origin = base = last_ts = last_t = pos = None
    dist = 0.0
    down_since = None          # 끊기기 전 마지막 프레임의 도착 시각
    down_dist = 0.0
    outages = []               # (끊기기 전 마지막 프레임 시각, 끊긴 시간 s, 끊기기 전 거리, 복구 때 거리)
    n = nbytes = 0
    max_gap = max_delay = 0.0
    tick = time.time() + 1.0
    try:
        while True:
            data = hub.get_latest(timeout=0.1)
            now = time.time()
            if data is not None:
                pkt.ParseFromString(data)
                if last_ts is not None and pkt.timestamp < last_ts - 1.0:
                    log("HL2 timestamp 가 되돌아갔다(앱 재시작). 출발점과 지연 기준을 다시 잡는다")
                    origin = base = None
                pos = (pkt.head_pos_x, pkt.head_pos_y, pkt.head_pos_z)
                if origin is None:
                    origin = pos
                    log(f"HL2DATA 수신 시작. 출발점 {fmt(pos)}")
                dist = math.hypot(pos[0] - origin[0], pos[2] - origin[2])
                d = now - pkt.timestamp
                base = d if base is None else min(base, d)
                if down_since is not None:
                    log(f"복구  {now - down_since:.1f} s 끊김. 지금 {dist:.1f} m {fmt(pos)}, "
                        f"지연 +{(d - base) * 1000:.0f} ms")
                    outages.append((down_since, now - down_since, down_dist, dist))
                    down_since = None
                if last_t is not None:
                    max_gap = max(max_gap, now - last_t)
                last_t, last_ts = now, pkt.timestamp
                n += 1
                nbytes += len(data)
                max_delay = max(max_delay, d - base)

            if last_t is not None and down_since is None and now - last_t > GAP_S:
                down_since, down_dist = last_t, dist
                log(f"끊김  {GAP_S:.0f} s 넘게 무수신. 마지막 프레임 {dist:.1f} m {fmt(pos)}")

            if now >= tick:
                if n:
                    log(f"{n:3d} fr {nbytes / 1024:6.0f} kB/s  최대 간격 {max_gap:.2f} s  "
                        f"지연 +{max_delay * 1000:4.0f} ms  위치 {fmt(pos)}  거리 {dist:.1f} m")
                elif last_t is not None:
                    log(f"  0 fr  무수신 {now - last_t:.1f} s  마지막 거리 {dist:.1f} m")
                n = nbytes = 0
                max_gap = max_delay = 0.0
                while tick <= now:
                    tick += 1.0
    except KeyboardInterrupt:
        pass
    finally:
        if down_since is not None:
            outages.append((down_since, None, down_dist, None))
        log(f"요약: 끊김 {len(outages)} 번")
        for t0, dur, d0, d1 in outages:
            if dur is None:
                log(f"  {datetime.fromtimestamp(t0):%H:%M:%S}  끝날 때까지 끊김  끊기기 전 {d0:.1f} m")
            else:
                log(f"  {datetime.fromtimestamp(t0):%H:%M:%S}  {dur:.1f} s  끊기기 전 {d0:.1f} m -> 복구 {d1:.1f} m")
        hub.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HL2 와 comm_hub 사이 연결이 어디서 끊기는지 기록한다")
    add_broker_args(ap)
    run(ap.parse_args())
