"""HL2 센서 스트림에서 사용자 상태를 정량화한다 — 실시간 시각화 + 실험용 녹화.

Adaptive-MM-UI 의 Server 를 이 저장소로 옮긴 것이다. 지표 정의는 그대로이고 배치만
바꿨다.

  획득  hl2ss 로 HL2 에 직접 연결한다. IMU 와 마이크는 comm_hub 가 나르는
        HL2SensorPacket 에 없으므로(기기 C# 앱이 담지 않는다) 이 경로가 필요하다.
  계산  modules/adaptivemm/ 의 StreamingMetrics 가 2 초 창으로 집계한다.
  배포  계산한 지표를 comm_hub 에 USER_STATE 로 UPLOAD 한다. 다른 진입점과 같은 규약이다.
  녹화  output/adaptivemm/<session>/ 에 이미지·이벤트는 append, 배열은 주기 flush.

실행:
    conda activate uvr_integ
    python comm_hub.py --port 37001          # 터미널 1 (지표를 구독할 쪽이 있을 때만)
    python main_adaptiveMM.py --hl2 <HL2_IP> # 터미널 2

    python main_adaptiveMM.py --hl2 <IP> --record          # 녹화까지
    python main_adaptiveMM.py --hl2 <IP> --no-hub          # 브로커 없이 단독
    python main_adaptiveMM.py --hl2 <IP> --no-gui          # 창 없이 (SSH)

창: 'PV'(+시선), 'Metrics'(지표 패널), 'Head'(top-down). q 또는 ESC 로 종료.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import zmq

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "_hl2ss"))
sys.path.insert(0, os.path.join(_ROOT, "_comm"))
import hl2ss                                                       # noqa: E402
import hl2ss_lnm                                                   # noqa: E402
from hub_client import HubClient, add_broker_args, KW_USER_STATE   # noqa: E402

sys.path.insert(0, _ROOT)
from modules.adaptivemm import si_adapter                          # noqa: E402
from modules.adaptivemm.audio import estimate_audio_density        # noqa: E402
from modules.adaptivemm.common import gaze_yaw_pitch               # noqa: E402
from modules.adaptivemm.head_view import render_head_view          # noqa: E402
from modules.adaptivemm.panel import render_metrics_panel          # noqa: E402
from modules.adaptivemm.streaming import StreamingMetrics          # noqa: E402
from modules.adaptivemm.visual import clutter_value, flow_residual, to_small  # noqa: E402

IDENTITY = b"ADAPTIVEMM"
RECORD_ROOT = os.path.join(_ROOT, "output", "adaptivemm")

PV_WIDTH, PV_HEIGHT, PV_FPS = 640, 360, 30
GAZE_DISTANCE = 1.5      # 시선 ray 를 찍을 고정 거리(m). 시선에는 깊이가 없다
QPC = 1e7                # HoloLens QPC ticks per second


# ─────────────────────────────────────────────────────────────────────────────
# 스트림 스레드. 각자 hl2ss 클라이언트 하나를 붙들고 지표에 push 한다.
# ─────────────────────────────────────────────────────────────────────────────
class _Stream(threading.Thread):
    """공통 뼈대. 자식은 open() 과 handle(packet) 만 구현한다."""

    name = "stream"

    def __init__(self, host, metrics, recorder=None):
        super().__init__(daemon=True)
        self.host, self.metrics, self.recorder = host, metrics, recorder
        self.client = None
        self.running = False
        self.n = 0
        self.error = None

    def open(self):
        raise NotImplementedError

    def handle(self, packet):
        raise NotImplementedError

    def run(self):
        try:
            self.client = self.open()
            self.client.open()
        except Exception as e:                       # 스트림 하나가 죽어도 나머지는 산다
            self.error = e
            print(f"[{self.name}] open 실패: {e}", flush=True)
            return
        self.running = True
        print(f"[{self.name}] 시작", flush=True)
        while self.running:
            try:
                packet = self.client.get_next_packet()
            except Exception as e:
                if self.running:
                    self.error = e
                    print(f"[{self.name}] 수신 중단: {e}", flush=True)
                break
            try:
                self.handle(packet)
                self.n += 1
            except Exception as e:
                print(f"[{self.name}] 처리 오류: {e}", flush=True)
        try:
            self.client.close()
        except Exception:
            pass
        print(f"[{self.name}] 종료 ({self.n} packets)", flush=True)

    def stop(self):
        self.running = False


class SpatialInput(_Stream):
    """head pose / eye ray / 양손 26 관절. gaze 와 hand 지표의 입력."""

    name = "SI"

    def __init__(self, host, metrics, recorder=None, trail_len=240):
        super().__init__(host, metrics, recorder)
        self.trail = deque(maxlen=trail_len)
        self.latest: si_adapter.SIFrame | None = None

    def open(self):
        return hl2ss_lnm.rx_si(self.host, hl2ss.StreamPort.SPATIAL_INPUT)

    def handle(self, packet):
        f = si_adapter.unpack(hl2ss, packet)
        self.latest = f
        ts = packet.timestamp

        if f.has_head:
            self.trail.append(f.head_position)
            if f.has_eye:
                yaw, pitch = gaze_yaw_pitch(f.eye_direction, f.head_forward, f.head_up)
                self.metrics.push_gaze(ts, yaw, pitch)

        for side in ("L", "R"):
            h = f.hand(side)
            if h is not None:
                # 손목 위치 하나로 손 전체의 운동을 대표한다 (haptic.py 와 같은 규약)
                self.metrics.push_hand(side, ts, h["position"][si_wrist()], True)
            else:
                self.metrics.push_hand(side, ts, (np.nan,) * 3, False)

        if self.recorder is not None:
            self.recorder.write_si(f)


def si_wrist() -> int:
    return hl2ss.SI_HandJointKind.Wrist


class Imu(_Stream):
    """가속도계 또는 자이로. 패킷마다 여러 샘플이 배치로 온다."""

    def __init__(self, host, metrics, port, kind, recorder=None):
        super().__init__(host, metrics, recorder)
        self.port, self.kind = port, kind
        self.name = f"IMU-{kind}"

    def open(self):
        return hl2ss_lnm.rx_rm_imu(self.host, self.port)

    def handle(self, packet):
        imu = hl2ss.unpack_rm_imu(packet.payload)
        xyz = np.stack([imu.x, imu.y, imu.z], axis=1).astype(np.float32)
        if self.kind == "accel":
            self.metrics.push_accel(packet.timestamp, xyz)
        else:
            self.metrics.push_gyro(packet.timestamp, xyz)
        if self.recorder is not None:
            self.recorder.write_imu(self.kind, packet.timestamp, xyz)


class Microphone(_Stream):
    """마이크. spectral flux 를 그 자리에서 계산해 audio_density 로 넣는다."""

    name = "MIC"

    def __init__(self, host, metrics, recorder=None):
        super().__init__(host, metrics, recorder)
        self._prev_spectrum = None
        self._opened_wav = False

    def open(self):
        return hl2ss_lnm.rx_microphone(self.host, hl2ss.StreamPort.MICROPHONE)

    def handle(self, packet):
        samples = packet.payload
        if not isinstance(samples, np.ndarray):
            return
        density, self._prev_spectrum = estimate_audio_density(samples, self._prev_spectrum)
        self.metrics.push_audio(packet.timestamp, density)
        if self.recorder is not None:
            if not self._opened_wav:
                self.recorder.open_audio(hl2ss.Parameters_MICROPHONE.CHANNELS,
                                         hl2ss.Parameters_MICROPHONE.SAMPLE_RATE)
                self._opened_wav = True
            self.recorder.write_audio(samples)


class PersonalVideo(_Stream):
    """PV 프레임. visual clutter 와 optical flow 의 입력이자 미리보기 화면.

    `--mrc` 를 켜면 홀로그램이 합성된 프레임이 온다. 화면에 무엇이 그려져 있는지까지
    포함해 시각 부하를 재고 싶을 때 쓴다.
    """

    name = "PV"

    def __init__(self, host, metrics, recorder=None, mrc=False, perspective=None):
        super().__init__(host, metrics, recorder)
        self.mrc = mrc
        self.perspective = perspective
        self._prev_small = None
        self.latest_bgr = None
        self.latest_ts = 0
        self._lock = threading.Lock()

    def open(self):
        # enable_mrc 를 켜면 같은 PV 스트림이 합성본으로 바뀐다. timestamp 와 intrinsics 가
        # 그대로라 아래 파이프라인은 손댈 것이 없다. (Device Portal 기반 rx_mrc 는 별도
        # 스트림이고 계정이 필요해 쓰지 않는다.)
        kw = dict(enable_mrc=self.mrc)
        if self.mrc and self.perspective is not None:
            kw["hologram_perspective"] = self.perspective
        hl2ss_lnm.start_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO, **kw)
        return hl2ss_lnm.rx_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO,
                               width=PV_WIDTH, height=PV_HEIGHT, framerate=PV_FPS)

    def handle(self, packet):
        bgr = packet.payload.image
        ts = packet.timestamp
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        self.metrics.push_clutter(ts, clutter_value(gray))
        small = to_small(gray)
        if self._prev_small is not None:
            self.metrics.push_flow(ts, flow_residual(self._prev_small, small))
        self._prev_small = small

        with self._lock:
            self.latest_bgr, self.latest_ts = bgr, ts
        if self.recorder is not None:
            self.recorder.write_pv(ts, bgr)

    def snapshot(self):
        with self._lock:
            return (None, 0) if self.latest_bgr is None else (self.latest_bgr.copy(), self.latest_ts)

    def stop(self):
        super().stop()
        try:
            hl2ss_lnm.stop_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
def draw_gaze(bgr, si_frame):
    """시선 ray 를 고정 거리에 찍어 PV 에 겹친다. 깊이가 없어 근사다."""
    if si_frame is None or not (si_frame.has_head and si_frame.has_eye):
        return bgr
    d = si_frame.eye_direction
    f, u = si_frame.head_forward, si_frame.head_up
    r = np.cross(f, u)
    fwd = float(np.dot(d, f))
    if fwd <= 1e-3:                                   # 뒤를 보면 찍지 않는다
        return bgr
    x = float(np.dot(d, r)) / fwd
    y = float(np.dot(d, u)) / fwd
    # 화각 근사: PV 수평 FOV 를 64도로 두고 정규화 좌표를 픽셀로
    fx = PV_WIDTH / (2 * np.tan(np.radians(64) / 2))
    u_px = int(PV_WIDTH / 2 + x * fx)
    v_px = int(PV_HEIGHT / 2 - y * fx)
    if 0 <= u_px < PV_WIDTH and 0 <= v_px < PV_HEIGHT:
        cv2.circle(bgr, (u_px, v_px), 9, (0, 255, 255), 2)
    return bgr


def main() -> None:
    ap = argparse.ArgumentParser(description="HL2 사용자 상태 정량화 (실시간 + 녹화)")
    ap.add_argument("--hl2", required=True, help="HoloLens2 IP (hl2ss 서버 주소)")
    add_broker_args(ap)
    ap.add_argument("--no-hub", action="store_true", help="comm_hub 로 지표를 보내지 않는다")
    ap.add_argument("--record", action="store_true", help="세션을 output/adaptivemm/ 에 녹화")
    ap.add_argument("--session", default=None, help="녹화 폴더 이름 (기본: 시각)")
    ap.add_argument("--no-gui", action="store_true", help="창을 띄우지 않는다")
    ap.add_argument("--mrc", action="store_true",
                    help="PV 를 홀로그램 합성 영상으로 받는다. visual_clutter/visual_flow 가 "
                         "실제 카메라 화면이 아니라 '사용자가 본 화면'을 재는 값이 된다")
    ap.add_argument("--hologram-perspective", choices=("pv", "display"), default="pv",
                    help="MRC 에서 홀로그램을 어느 시점으로 렌더링할지. pv = PV 카메라 시점"
                         "(카메라 영상과 정합), display = 사용자 눈 시점. 기본 pv")
    ap.add_argument("--window", type=float, default=2.0, help="지표 집계 창 (초)")
    ap.add_argument("--rate", type=float, default=10.0, help="지표 계산·전송 주기 (Hz)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-imu", action="store_true")
    args = ap.parse_args()

    metrics = StreamingMetrics(window=args.window)

    recorder = None
    if args.record:
        from modules.adaptivemm.recorder import SessionRecorder
        session = args.session or datetime.now().strftime("sess_%Y%m%d_%H%M%S")
        recorder = SessionRecorder(RECORD_ROOT, session, meta={
            "host": args.hl2, "pv": [PV_WIDTH, PV_HEIGHT, PV_FPS], "mrc": args.mrc,
            "hologram_perspective": args.hologram_perspective if args.mrc else None,
            "window_s": args.window, "rate_hz": args.rate})
        print(f"녹화 -> {recorder.dir}", flush=True)

    hub = None
    if not args.no_hub:
        try:
            hub = HubClient(args.host, args.port, result_kw=KW_USER_STATE, identity=IDENTITY)
            # ZeroMQ 는 상대가 없어도 connect 가 성공하고 전송분을 큐에 쌓는다. 큐가 차면
            # send 가 블록되어 루프 전체가 멈춘다. 브로커 없이 단독 실행하는 경우가 있으므로
            # 전송에 시한을 두고, 넘치면 그 tick 을 버린다.
            hub.tx.setsockopt(zmq.SNDTIMEO, 20)     # tick 주기를 잡아먹지 않을 만큼만 기다린다
            hub.tx.setsockopt(zmq.SNDHWM, 100)
        except Exception as e:
            print(f"comm_hub 연결 실패, 지표 전송 없이 진행한다: {e}", flush=True)
            hub = None

    perspective = (hl2ss.HologramPerspective.PV if args.hologram_perspective == "pv"
                   else hl2ss.HologramPerspective.DISPLAY)
    pv = PersonalVideo(args.hl2, metrics, recorder, mrc=args.mrc, perspective=perspective)
    si = SpatialInput(args.hl2, metrics, recorder)
    streams: list[_Stream] = [pv, si]
    if not args.no_imu:
        streams += [Imu(args.hl2, metrics, hl2ss.StreamPort.RM_IMU_ACCELEROMETER, "accel", recorder),
                    Imu(args.hl2, metrics, hl2ss.StreamPort.RM_IMU_GYROSCOPE, "gyro", recorder)]
    if not args.no_audio:
        streams.append(Microphone(args.hl2, metrics, recorder))
    for s in streams:
        s.start()

    print(f"지표 창 {args.window}s, 전송 {args.rate} Hz. q 또는 ESC 로 종료.", flush=True)
    period = 1.0 / args.rate
    next_tick = time.time()
    last_log = time.time()
    ticks = dropped = 0

    try:
        while True:
            now_qpc = int(time.time() * QPC)          # 표시용. 지표는 패킷 QPC 로 판단한다
            values = metrics.current(now_qpc)
            ticks += 1

            if hub is not None:
                payload = json.dumps({"timestamp": now_qpc,
                                      **{k: (None if v is None or np.isnan(v) else round(float(v), 6))
                                         for k, v in values.items()}}).encode()
                try:
                    hub.send(payload)
                except zmq.Again:                    # 브로커가 없거나 밀린다. 이 tick 은 버린다
                    if not dropped:
                        print("comm_hub 가 받지 않는다. 지표 전송을 건너뛴다 "
                              "(--no-hub 로 끌 수 있다)", flush=True)
                    dropped += 1
            if recorder is not None:
                recorder.write_metrics(now_qpc, values)

            if not args.no_gui:
                bgr, _ = pv.snapshot()
                if bgr is not None:
                    cv2.imshow("PV", draw_gaze(bgr, si.latest))
                cv2.imshow("Metrics", render_metrics_panel(values))
                f = si.latest
                if f is not None and f.has_head:
                    cv2.imshow("Head", render_head_view(f.head_position, f.head_forward,
                                                        trail=list(si.trail)))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            if time.time() - last_log >= 5.0:
                alive = ", ".join(f"{s.name}:{s.n}" for s in streams if s.running)
                drop = f"  hub_dropped={dropped}" if dropped else ""
                print(f"  ticks={ticks}  {alive}{drop}", flush=True)
                last_log = time.time()

            next_tick += period
            time.sleep(max(0.0, next_tick - time.time()))
    except KeyboardInterrupt:
        print("\n중단", flush=True)
    finally:
        for s in streams:
            s.stop()
        for s in streams:
            s.join(timeout=2.0)
        if hub is not None:
            hub.close()
        if recorder is not None:
            print("녹화 요약:", json.dumps(recorder.close(), ensure_ascii=False), flush=True)
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
