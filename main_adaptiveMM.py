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
import traceback
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
import hl2_data_pb2 as hl2proto                                   # noqa: E402
import hl2_sensors_pb2 as sensproto                               # noqa: E402
from hub_client import (                                          # noqa: E402
    HubClient, add_broker_args, KW_USER_STATE,
    KW_HL2DATA, KW_HL2_AUDIO, KW_HL2_CONTROL, KW_HL2_RENDER)

sys.path.insert(0, _ROOT)
from modules.adaptivemm import si_adapter, timebase as TB          # noqa: E402
from modules.adaptivemm.headmotion import HeadMotion               # noqa: E402
from modules.adaptivemm.audio import estimate_audio_density        # noqa: E402
from modules.adaptivemm.common import gaze_yaw_pitch               # noqa: E402
from modules.adaptivemm.head_view import render_head_view          # noqa: E402
from modules.adaptivemm.panel import render_metrics_panel          # noqa: E402
from modules.adaptivemm.streaming import StreamingMetrics          # noqa: E402
from modules.adaptivemm.visual import clutter_value, flow_residual, to_small  # noqa: E402

IDENTITY = b"ADAPTIVEMM"
RECORD_ROOT = os.path.join(_ROOT, "output", "adaptivemm")

PV_WIDTH, PV_HEIGHT, PV_FPS = 640, 360, 30
CONTROL_PERIOD_S = 5.0   # 기기가 이 주기로 제어를 못 받으면 스트림을 놓는다(기기 타임아웃 15 s)
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


class _AudioProbe:
    """audio_density 는 무음이 들어와도 청크가 절반씩 빠져도 계속 갱신된다. 그래서 따로 잰다.

        samples/s 대 sample_rate   100% 에서 벗어나면 청크가 빠지거나 rate 신고가 틀렸다
        level (dBFS)               마이크가 죽었는지
        lag                        오디오 timestamp 가 다른 스트림과 같은 축인지.
                                   일정하면 정상(마이크 파이프라인 지연), 커지면 축이 다르다
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.sample_rate = self.channels = 0
        self.last_ts = None
        self._zero()

    def _zero(self):
        self.chunks = self.samples = 0
        self.sumsq = 0.0
        self.peak = 0.0

    def observe(self, pcm, ts, sample_rate, channels):
        """채널 병합 전 원본을 넣는다. 프레임 수를 세려면 신고된 채널 수가 필요하다."""
        x = np.asarray(pcm).ravel()
        if x.size == 0:
            return
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
        with self._lock:
            self.chunks += 1
            self.samples += x.size
            self.sumsq += float(np.dot(x, x))
            self.peak = max(self.peak, float(np.abs(x).max()))
            self.sample_rate, self.channels = int(sample_rate), int(channels)
            self.last_ts = ts

    def report(self, now_qpc, elapsed):
        with self._lock:
            n, s, sq, pk = self.chunks, self.samples, self.sumsq, self.peak
            sr, ch, last = self.sample_rate, self.channels, self.last_ts
            self._zero()
        if n == 0:
            return "  audio    수신 없음"
        db = lambda v: f"{20 * np.log10(v):6.1f}" if v > 1e-9 else "  -inf"
        fps = s / max(1, ch) / elapsed                 # 초당 프레임(채널 병합 기준)
        cover = f"{fps / sr * 100:5.1f}%" if sr else "    ?%"
        lag = (now_qpc - last) / QPC * 1e3
        return (f"  audio    {sr}Hz/{ch}ch  {n / elapsed:4.1f}chunk/s  "
                f"{fps:6.0f}frame/s = {cover} of rate  lag {lag:7.1f}ms  "
                f"rms {db(s and (sq / s) ** 0.5)} peak {db(pk)} dBFS")


class Microphone(_Stream):
    """마이크. spectral flux 를 그 자리에서 계산해 audio_density 로 넣는다."""

    name = "MIC"

    def __init__(self, host, metrics, recorder=None):
        super().__init__(host, metrics, recorder)
        self._prev_spectrum = None
        self._opened_wav = False
        self.audio_probe = _AudioProbe()

    def open(self):
        return hl2ss_lnm.rx_microphone(self.host, hl2ss.StreamPort.MICROPHONE)

    def handle(self, packet):
        samples = packet.payload
        if not isinstance(samples, np.ndarray):
            return
        self.audio_probe.observe(samples, packet.timestamp,
                                 hl2ss.Parameters_MICROPHONE.SAMPLE_RATE,
                                 hl2ss.Parameters_MICROPHONE.CHANNELS)
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

        edge, _tex = clutter_value(gray)                 # (edge_density, texture_energy)
        self.metrics.push_clutter(ts, edge)
        small = to_small(gray)
        if self._prev_small is not None:
            res, _ego = flow_residual(self._prev_small, small)
            self.metrics.push_flow(ts, res)
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
# comm_hub 경로. 기기 C# 앱이 올리는 것을 구독한다.
#
# hl2ss 경로와 다른 점 셋.
#   - 시각이 Unity Time.time(초) / 오디오는 ns 다. 입구에서 QPC 로 올린다.
#   - IMU 가 없다. head pose 를 미분해 선형가속도와 각속도를 만든다(중력 없음).
#   - 오디오가 int16 PCM 이다. audio.estimate_audio_density 가 dtype 을 보고 정규화한다.
# ─────────────────────────────────────────────────────────────────────────────
# 레이어를 기다리는 시간. 프레임 개수로 세면 안 된다 - PV 가 30 Hz 에서 10 Hz 로
# 떨어지면 같은 개수가 전혀 다른 시간이 되고, 느릴 때일수록 덜 기다리게 된다.
RENDER_WAIT_S = 0.6


class _RenderStats:
    """AR 레이어가 실제로 얼마나 오고 몇 프레임이 짝을 찾는지.

    도달률은 기기에 물어볼 것이 아니라 여기서 재면 된다. 짝을 못 찾은 프레임은 raw 로
    계산되므로, unpaired 가 많으면 '화면 기준' 이 아니라 '카메라 기준' 지표를 보고 있는 것이다.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.total_arrived = self.total_paired = 0
        self._zero()

    def _zero(self):
        self.arrived = self.nbytes = self.paired = self.unpaired = 0
        self.late = self.nomatch = 0
        self.delays = []

    def on_late(self):
        with self._lock:
            self.late += 1

    def on_nomatch(self):
        with self._lock:
            self.nomatch += 1

    def on_delay(self, dt):
        with self._lock:
            self.delays.append(dt)

    def on_arrival(self, nbytes):
        with self._lock:
            self.arrived += 1
            self.nbytes += nbytes
            self.total_arrived += 1

    def on_frame(self, paired):
        with self._lock:
            if paired:
                self.paired += 1
                self.total_paired += 1
            else:
                self.unpaired += 1

    def status(self, elapsed, pv_rate=0.0):
        """평상시 로그용. (한 줄 요약, 경고 또는 None).

        '안 온다' 와 '와도 짝이 안 맞는다' 는 둘 다 조용히 raw 로 떨어져 증상이 같다.
        원인이 다르므로 구분해서 알린다.
        """
        with self._lock:
            a, pr, u = self.arrived, self.paired, self.unpaired
            ta, late, nm = self.total_arrived, self.late, self.nomatch
            d = sorted(self.delays)
        tot = pr + u
        if a == 0:
            return ("render 없음",
                    "AR 레이어가 오지 않는다. 기기가 HL2_CONTROL(send_render) 을 받았는지, "
                    "그 빌드에 레이어 전송이 들어갔는지 확인하라. 지금은 raw 로 재고 있다."
                    if ta == 0 else
                    "AR 레이어가 끊겼다. 지금은 raw 로 재고 있다.")
        rate = pr / tot * 100 if tot else 0.0
        line = f"render {a / elapsed:.1f}/s paired {rate:.0f}%"
        if d:
            line += f" delay {d[len(d) // 2] * 1e3:.0f}/{d[-1] * 1e3:.0f}ms"   # 중앙값/최대
        if not tot or rate >= 50:
            return (line, None)
        if nm > late:
            return (line,
                    "레이어 timestamp 가 어느 PV 프레임과도 맞지 않는다. 기기가 "
                    "HL2SensorPacket.timestamp 를 그대로 echo 하는지 확인하라.")
        # timestamp 는 맞는데 그 프레임을 이미 내보낸 뒤다.
        if pv_rate > 0 and a < pv_rate * 0.9:
            return (line,
                    f"레이어가 PV 보다 적게 온다 ({a / elapsed:.1f}/s vs {pv_rate:.1f}/s). "
                    "밀린 만큼 계속 벌어지므로 대기를 늘려도 못 따라잡는다. 레이어 크기나 "
                    "전송 주기를 줄여야 한다.")
        return (line,
                f"레이어가 대기 시간({RENDER_WAIT_S}s)보다 늦게 온다. RENDER_WAIT_S 를 늘리면 된다.")

    def roll(self):
        """블록 경계. status 와 report 가 같은 구간을 보도록 리셋은 여기서만 한다."""
        with self._lock:
            self._zero()

    def report(self, elapsed):
        with self._lock:
            a, b, pr, u = self.arrived, self.nbytes, self.paired, self.unpaired
        tot = pr + u
        if a == 0 and tot == 0:
            return "  render   수신 없음"
        rate = f"{pr / tot * 100:5.1f}%" if tot else "     -"
        return (f"  render   {a / elapsed:4.1f}layer/s  {b / elapsed / 1024:6.1f}KB/s  "
                f"{b / max(1, a) / 1024:5.1f}KB/layer  paired {rate} ({pr}/{tot})")


class HubSource(threading.Thread):
    """HL2DATA(+HL2_AUDIO) 를 구독해 지표에 넣는다. 스트림 스레드들을 대체한다."""

    name = "HUB"

    def __init__(self, host, port, metrics, recorder=None, want_audio=True,
                 want_render=False):
        super().__init__(daemon=True)
        self.metrics, self.recorder = metrics, recorder
        self.running = False
        self.n = self.n_audio = 0
        self.trail = deque(maxlen=240)
        self.latest: si_adapter.SIFrame | None = None
        self.latest_bgr, self.latest_ts = None, 0
        # raw 는 도착 즉시, 합성본은 레이어를 기다린 뒤 갱신된다. 합성본이 조금 늦다.
        self.latest_comp = None
        self._lock = threading.Lock()
        self._prev_small = None
        self._prev_spectrum = None
        self._head = HeadMotion()
        self._wav_open = False
        self.want_audio = want_audio
        self.want_render = want_render
        self.audio_probe = _AudioProbe()
        self.render_stats = _RenderStats()
        # 레이어가 늦게 오므로 PV 를 조금 붙들었다가 짝을 맞춰 처리한다. 즉시 처리하면
        # 짝이 항상 비어 있어 레이어가 한 번도 쓰이지 않는다.
        self._vis_lock = threading.Lock()
        self._pending = deque()          # (ts_float, ts_qpc, bgr). 도착 순 = timestamp 순
        self._layers = {}                # ts_float -> (bgra, png bytes)
        self._last_vis_ts = None         # 이미 처리한 가장 최근 프레임 시각(초)
        self._alpha_checked = False
        self._render_logged = False
        self._size_warned = False
        self.miss = None                 # 짝이 안 맞을 때 마지막 표본. 경고에 같이 찍는다

        # comm_hub 는 source 단위로 구독을 덮어쓰므로 keyword 마다 identity 를 나눈다.
        self.data = HubClient(host, port, recv_kw=KW_HL2DATA,
                              result_kw=KW_USER_STATE, identity=IDENTITY)
        self.data.tx.setsockopt(zmq.SNDTIMEO, 20)
        self.data.tx.setsockopt(zmq.SNDHWM, 100)
        self.audio = None
        if want_audio:
            # 오디오는 conflate 하면 안 된다. 청크가 빠지면 spectral flux 의 연속성이 깨진다.
            self.audio = HubClient(host, port, recv_kw=KW_HL2_AUDIO,
                                   identity=IDENTITY + b"_AUD", queue_size=64)
        self.render = None
        if want_render:
            # 레이어도 conflate 하면 안 된다. 빠진 프레임은 raw 로 떨어져 지표가 섞인다.
            self.render = HubClient(host, port, recv_kw=KW_HL2_RENDER,
                                    identity=IDENTITY + b"_RND", queue_size=16)

    # --- 기기 제어 -------------------------------------------------------
    def send_control(self, send_audio: bool, send_render: bool = False) -> None:
        msg = sensproto.HL2Control(send_audio=send_audio, send_imu=False,
                                   send_render=send_render)
        try:
            self.data.send(msg.SerializeToString(), kw=KW_HL2_CONTROL)
        except zmq.Again:
            pass

    # --- 루프 ------------------------------------------------------------
    def run(self):
        self.running = True
        print("[HUB] 구독 시작", flush=True)
        if self.want_audio:
            threading.Thread(target=self._audio_loop, daemon=True).start()
        if self.want_render:
            threading.Thread(target=self._render_loop, daemon=True).start()
        while self.running:
            buf = self.data.get_latest(timeout=0.5)
            if buf is None:
                continue
            try:
                self._on_packet(buf)
                self.n += 1
            except Exception:
                print("[HUB] 패킷 처리 오류\n" + traceback.format_exc(), flush=True)
        print(f"[HUB] 종료 ({self.n} packets, audio {self.n_audio})", flush=True)

    def _audio_loop(self):
        while self.running:
            buf = self.audio.get_latest(timeout=0.5)
            if buf is None:
                continue
            try:
                self._on_audio(buf)
                self.n_audio += 1
            except Exception:
                print("[HUB] 오디오 오류\n" + traceback.format_exc(), flush=True)

    # --- HL2DATA ---------------------------------------------------------
    def _on_packet(self, buf):
        pkt = hl2proto.HL2SensorPacket()
        pkt.ParseFromString(buf)
        ts = TB.from_seconds(pkt.timestamp)          # Time.time 초 -> QPC

        # 영상: clutter 와 optical flow
        if pkt.image_data:
            arr = np.frombuffer(pkt.image_data, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                with self._lock:
                    self.latest_bgr, self.latest_ts = bgr, ts
                with self._vis_lock:
                    self._pending.append((float(pkt.timestamp), ts, bgr, time.time()))
                    self._drain()

        # head 운동: IMU 가 없으므로 pose 를 미분한다. 나온 값에는 중력이 없다.
        pos = (pkt.head_pos_x, pkt.head_pos_y, pkt.head_pos_z)
        rot = (pkt.head_rot_x, pkt.head_rot_y, pkt.head_rot_z, pkt.head_rot_w)
        d = self._head.push(float(pkt.timestamp), pos, rot)
        if d is not None:
            lin, ang = d
            self.metrics.push_accel(ts, lin[None, :])
            self.metrics.push_gyro(ts, ang[None, :])

        # SI: 시선과 손. 기기가 쿼터니언으로 주므로 forward/up 을 만들어 쓴다.
        f = si_adapter.SIFrame(ts)
        f.head_position = np.array(pos, np.float32)
        fwd, up = _quat_axes(rot)
        f.head_forward, f.head_up = fwd, up
        if any((pkt.eye_gaze_dir_x, pkt.eye_gaze_dir_y, pkt.eye_gaze_dir_z)):
            f.eye_origin = np.array((pkt.eye_gaze_origin_x, pkt.eye_gaze_origin_y,
                                     pkt.eye_gaze_origin_z), np.float32)
            f.eye_direction = np.array((pkt.eye_gaze_dir_x, pkt.eye_gaze_dir_y,
                                        pkt.eye_gaze_dir_z), np.float32)
            yaw, pitch = gaze_yaw_pitch(f.eye_direction, fwd, up)
            self.metrics.push_gaze(ts, yaw, pitch)

        for side, joints in (("L", pkt.left_hand), ("R", pkt.right_hand)):
            if len(joints) >= 3:
                P = np.asarray(joints, np.float32).reshape(-1, 3)
                self.metrics.push_hand(side, ts, P[0], True)
                h = {"position": P, "orientation": np.zeros((len(P), 4), np.float32)}
                if side == "L":
                    f.hand_left = h
                else:
                    f.hand_right = h
            else:
                self.metrics.push_hand(side, ts, (np.nan,) * 3, False)

        self.trail.append(f.head_position)
        self.latest = f
        if self.recorder is not None:
            self.recorder.write_si(f)

    # --- 영상: 레이어 짝짓기와 합성 --------------------------------------
    def _drain(self):
        """_vis_lock 을 쥔 채로 부른다. 짝이 붙었거나 기다릴 만큼 기다린 것부터 처리한다.

        왼쪽에서만 꺼낸다. 프레임 2 의 레이어가 프레임 1 보다 먼저 와도 순서를 지켜야
        optical flow 가 이웃 프레임 쌍을 보게 된다.
        """
        while self._pending:
            ts_f, ts_q, bgr, enq = self._pending[0]
            layer = self._layers.pop(ts_f, None)
            if layer is None and self.want_render and time.time() - enq < RENDER_WAIT_S:
                break                                  # 아직 레이어를 기다릴 여지가 있다
            self._pending.popleft()
            self._last_vis_ts = ts_f
            # 처리한 시각보다 오래된 레이어는 짝을 잃었다. 두면 계속 쌓인다.
            for k in [k for k in self._layers if k <= ts_f]:
                del self._layers[k]
            self._process_visual(ts_q, bgr, layer)

    def _process_visual(self, ts, bgr, layer):
        if layer is not None:
            bgra, png = layer
            if bgra.shape[:2] != bgr.shape[:2] and not self._size_warned:
                self._size_warned = True
                print(f"[HUB] 레이어 격자가 raw 와 다르다: 레이어 {bgra.shape[1]}x{bgra.shape[0]} "
                      f"vs raw {bgr.shape[1]}x{bgr.shape[0]}. resize 해서 겹치므로 정합이 "
                      "어긋날 수 있다.", flush=True)
            frame = self._composite(bgr, bgra)
            if self.recorder is not None:
                self.recorder.write_render(ts, png)
        else:
            frame = bgr
        self.render_stats.on_frame(layer is not None)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge, _tex = clutter_value(gray)                # (edge_density, texture_energy)
        self.metrics.push_clutter(ts, edge)
        small = to_small(gray)
        if self._prev_small is not None:
            res, _ego = flow_residual(self._prev_small, small)
            self.metrics.push_flow(ts, res)
        self._prev_small = small
        if layer is not None:
            with self._lock:
                self.latest_comp = frame
        if self.recorder is not None:
            # 원본을 남긴다. 합성본은 raw 와 레이어로 언제든 다시 만들 수 있다.
            self.recorder.write_pv(ts, bgr)

    @staticmethod
    def _composite(bgr, bgra):
        """레이어를 raw 위에 알파 합성한다.

        cv2.imdecode 는 PNG 내부가 RGBA 여도 BGRA 로 돌려준다. 여기에 RGB2BGR 을 걸면
        R 과 B 가 뒤집힌다. 채널 변환 없이 그대로 쓰는 것이 맞다.
        """
        if bgra.ndim != 3 or bgra.shape[2] != 4:
            return bgr
        if bgra.shape[:2] != bgr.shape[:2]:
            bgra = cv2.resize(bgra, (bgr.shape[1], bgr.shape[0]), cv2.INTER_NEAREST)
        a = bgra[:, :, 3:4].astype(np.float32) / 255.0
        fg = bgra[:, :, :3].astype(np.float32)
        return (fg * a + bgr.astype(np.float32) * (1.0 - a)).astype(np.uint8)

    def _check_alpha_once(self, bgra):
        """첫 레이어에서 알파를 확인해 한 번만 찍는다.

        투명 비율이 먼저다. 배경이 투명하지 않으면 합성본이 레이어로 덮여 raw 가 사라진다.
        straight/premultiplied 는 색이 항상 알파 이하인지로 가른다.
        """
        a = bgra[:, :, 3]
        tot = a.size
        clear = float((a == 0).mean())
        solid = float((a == 255).mean())
        print(f"[HUB] AR 레이어 알파 분포: 투명 {clear*100:.1f}% / "
              f"반투명 {(1-clear-solid)*100:.1f}% / 불투명 {solid*100:.1f}%", flush=True)
        if clear < 0.5:
            print("       배경이 투명하지 않다. 합성본이 레이어로 덮여 raw 가 보이지 않는다. "
                  "기기가 배경을 알파 0 으로 비우는지 확인이 필요하다.", flush=True)
        m = (a > 8) & (a < 248)
        n = int(m.sum())
        if n < 50:
            return                                     # 반투명 화소가 없다. 다음 레이어에서 다시
        self._alpha_checked = True
        fg = bgra[:, :, :3][m].max(axis=1).astype(np.int32)
        frac = float((fg > a[m].astype(np.int32) + 2).mean())
        if frac > 0.02:
            print(f"[HUB] AR 레이어 알파: straight (반투명 {n}화소, 색>알파 {frac*100:.1f}%). "
                  "지금 합성식이 맞다.", flush=True)
        else:
            print(f"[HUB] AR 레이어 알파: premultiplied 로 보인다 (반투명 {n}화소, "
                  f"색>알파 {frac*100:.1f}%). 합성식을 fg + bg*(1-a) 로 바꿔야 "
                  "홀로그램 가장자리에 검은 테가 안 생긴다.", flush=True)

    # --- HL2_RENDER ------------------------------------------------------
    def _render_loop(self):
        while self.running:
            buf = self.render.get_latest(timeout=0.5)
            if buf is None:
                continue
            try:
                self._on_render(buf)
            except Exception:
                print("[HUB] 레이어 처리 오류\n" + traceback.format_exc(), flush=True)

    def _on_render(self, buf):
        r = sensproto.HL2Render()
        r.ParseFromString(buf)
        if not r.image:
            return
        self.render_stats.on_arrival(len(r.image))
        bgra = cv2.imdecode(np.frombuffer(r.image, np.uint8), cv2.IMREAD_UNCHANGED)
        if bgra is None:
            return
        if not self._render_logged:
            self._render_logged = True
            ch = bgra.shape[2] if bgra.ndim == 3 else 1
            print(f"[HUB] AR 레이어 수신 시작: {bgra.shape[1]}x{bgra.shape[0]} "
                  f"{ch}ch, {len(r.image) / 1024:.1f}KB", flush=True)
        if not self._alpha_checked:
            self._check_alpha_once(bgra)
        # 짝짓기 키는 기기가 그대로 echo 한 float 이다. 양쪽 다 같은 float32 비트라
        # 그대로 비교하면 정확히 맞는다. 반올림은 경계에서 갈릴 위험만 더한다.
        ts_f = float(r.timestamp)
        now = time.time()
        with self._vis_lock:
            hit = next((x for x in self._pending if x[0] == ts_f), None)
            if hit is not None:
                self.render_stats.on_delay(now - hit[3])
            elif self._last_vis_ts is not None and ts_f <= self._last_vis_ts:
                # timestamp 는 맞지만 그 프레임을 이미 내보냈다. 대기 시간이 모자란 것이다.
                self.render_stats.on_late()
                self.miss = (f"레이어 ts={ts_f:.4f} 가 이미 처리한 "
                             f"ts={self._last_vis_ts:.4f} 뒤에 왔다. 대기 {RENDER_WAIT_S}s 초과.")
                return
            else:
                self.render_stats.on_nomatch()
                self._note_mismatch(ts_f)
            self._layers[ts_f] = (bgra, bytes(r.image))
            self._drain()

    def _note_mismatch(self, ts_f):
        """_vis_lock 을 쥔 채로 부른다. 짝이 없을 때 가장 가까운 PV 와의 차이를 남긴다."""
        if not self._pending:
            self.miss = f"레이어 ts={ts_f:.4f} 도착 시 대기 중인 PV 프레임이 없다."
            return
        near = min(self._pending, key=lambda x: abs(x[0] - ts_f))[0]
        lo, hi = self._pending[0][0], self._pending[-1][0]
        self.miss = (f"레이어 ts={ts_f:.4f}, 가장 가까운 PV ts={near:.4f} "
                     f"(차이 {(ts_f - near) * 1e3:+.1f}ms). 대기열 {len(self._pending)}개 "
                     f"[{lo:.4f}~{hi:.4f}]")

    # --- HL2_AUDIO -------------------------------------------------------
    def _on_audio(self, buf):
        a = sensproto.HL2Audio()
        a.ParseFromString(buf)
        if a.is_aac:
            return                                    # 기기는 현재 raw PCM 만 보낸다
        pcm = np.frombuffer(a.audio, "<i2")            # int16 little-endian
        if pcm.size == 0:
            return
        ts = TB.from_nanoseconds(a.start_timestamp)    # ns -> QPC
        self.audio_probe.observe(pcm, ts, a.sample_rate, max(1, a.channels))
        if a.channels > 1:
            pcm = pcm.reshape(-1, a.channels).mean(axis=1).astype(np.int16)
        # 창 길이를 청크에 맞춘다. 512 샘플에 1024 창을 쓰면 절반이 zero padding 된다.
        n = 1 << max(6, int(pcm.size).bit_length() - 1)
        density, self._prev_spectrum = estimate_audio_density(
            pcm, self._prev_spectrum, window_size=n)
        self.metrics.push_audio(ts, density)
        if self.recorder is not None:
            if not self._wav_open:
                self.recorder.open_audio(1, a.sample_rate or 16000, sampwidth=2)
                self._wav_open = True
            self.recorder.write_audio_raw(pcm.tobytes())

    def send(self, payload: bytes) -> None:
        """지표 업로드. _loop 가 hub 와 같은 인터페이스로 쓴다."""
        self.data.send(payload)

    def snapshot(self):
        with self._lock:
            return (None, 0) if self.latest_bgr is None else (self.latest_bgr.copy(), self.latest_ts)

    def snapshot_comp(self):
        """레이어가 붙은 프레임의 합성본. 없으면 None."""
        with self._lock:
            return None if self.latest_comp is None else self.latest_comp.copy()

    def stop(self):
        self.running = False

    def close(self):
        self.data.close()
        if self.audio is not None:
            self.audio.close()
        if self.render is not None:
            self.render.close()


def _quat_axes(q):
    """(x, y, z, w) -> (forward, up). Unity 규약대로 +Z 가 forward, +Y 가 up."""
    x, y, z, w = (float(v) for v in q)
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-6:
        return np.array([0, 0, 1], np.float32), np.array([0, 1, 0], np.float32)
    x, y, z, w = x / n, y / n, z / n, w / n
    fwd = np.array([2 * (x * z + w * y), 2 * (y * z - w * x),
                    1 - 2 * (x * x + y * y)], np.float32)
    up = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z),
                   2 * (y * z + w * x)], np.float32)
    return fwd, up


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
    ap.add_argument("--source", choices=("hub", "hl2ss"), default="hub",
                    help="hub = 기기 C# 앱이 comm_hub 로 올리는 것을 구독 (기본). "
                         "hl2ss = HL2 에 직접 연결. 진짜 IMU 를 쓸 수 있으나 앱이 따로 필요하다")
    ap.add_argument("--hl2", default=None, help="HoloLens2 IP. --source hl2ss 에서만 쓴다")
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
    ap.add_argument("--no-render", action="store_true",
                    help="AR 레이어를 요청하지 않고 raw PV 로만 visual_clutter/visual_flow 를 "
                         "잰다. 기본은 레이어를 받아 합성한 '사용자가 본 화면' 기준이다")
    ap.add_argument("--diag", action="store_true",
                    help="5 s 마다 스트림 수신율·오디오 상태·지표별 창 샘플 수와 변동폭을 찍는다. "
                         "값이 갱신되는 것과 제대로 들어오는 것을 구분할 때 쓴다")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-imu", action="store_true")
    args = ap.parse_args()

    if args.source == "hl2ss" and not args.hl2:
        ap.error("--source hl2ss 에는 --hl2 <IP> 가 필요하다")
    if args.source == "hub" and args.mrc:
        # 조용히 무시하면 원본 카메라 영상을 '사용자가 본 화면' 으로 착각한 채 재게 된다.
        ap.error("--mrc 는 --source hl2ss 전용이다. hub 경로의 화면은 기기 앱이 "
                 "image_data 에 담아 보내는 것이고, 서버는 합성에 관여하지 않는다")

    # head pose 미분값에는 중력이 없다. hl2ss 의 가속도계 원본에는 있다.
    metrics = StreamingMetrics(window=args.window,
                               accel_has_gravity=(args.source == "hl2ss"))

    recorder = None
    if args.record:
        from modules.adaptivemm.recorder import SessionRecorder
        session = args.session or datetime.now().strftime("sess_%Y%m%d_%H%M%S")
        recorder = SessionRecorder(RECORD_ROOT, session, meta={
            "host": args.hl2, "pv": [PV_WIDTH, PV_HEIGHT, PV_FPS], "mrc": args.mrc,
            "hologram_perspective": args.hologram_perspective if args.mrc else None,
            "window_s": args.window, "rate_hz": args.rate})
        print(f"녹화 -> {recorder.dir}", flush=True)

    if args.source == "hub":
        src = HubSource(args.host, args.port, metrics, recorder,
                        want_audio=not args.no_audio, want_render=not args.no_render)
        src.start()
        src.send_control(send_audio=not args.no_audio, send_render=not args.no_render)
        streams = [src]
        pv = si = src
        hub = None                                   # 지표 업로드는 src 가 겸한다
        print(f"comm_hub {args.host}:{args.port} 구독. 제어 재전송 {CONTROL_PERIOD_S}s",
              flush=True)
        return _loop(args, metrics, recorder, streams, pv, si, src)

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
    _loop(args, metrics, recorder, streams, pv, si, hub)



# 지표 키 -> occupancy 키. gaze 두 지표는 같은 deque 에서 나온다.
_OCC_KEY = {"visual_clutter": "visual_clutter", "visual_flow": "visual_flow",
            "audio_density": "audio_density", "gaze_concentration": "gaze",
            "gaze_spread": "gaze", "head_lin_acc": "head_lin_acc",
            "head_ang_vel": "head_ang_vel", "hand_acc_L": "hand_L", "hand_acc_R": "hand_R"}


def _print_diag(metrics, streams, probe, now_qpc, elapsed, span, prev_n) -> None:
    """지표가 '움직인다' 를 넘어 '맞게 들어온다' 를 볼 수 있는 만큼만 찍는다.

    n_win 이 0 이면 그 지표는 None 이고, 몇 개뿐이면 값이 나와도 믿을 것이 못 된다.
    min 과 max 가 같으면 값이 갱신되는 것처럼 보여도 실은 고정돼 있다는 뜻이다.
    """
    rate = "  ".join(f"{s.name} {(s.n - prev_n.get(s.name, 0)) / elapsed:5.1f}/s"
                     for s in streams if getattr(s, "running", False))
    print(f"  [diag] {elapsed:.1f}s   {rate}", flush=True)
    if probe is not None:
        print(probe.report(now_qpc, elapsed), flush=True)
    src = next((s for s in streams if getattr(s, "want_render", False)), None)
    if src is not None:
        # queue-drop 이 0 이면 손실은 이쪽이 아니라 전송 구간이다.
        c = src.render
        print(src.render_stats.report(elapsed)
              + f"  socket {c.n_arrived} recv / {c.n_dropped} queue-drop", flush=True)
    occ = metrics.occupancy(now_qpc)
    print(f"  {'metric':<20}{'n_win':>7}{'min':>11}{'max':>11}", flush=True)
    for k, ok in _OCC_KEY.items():
        n = occ[ok]
        lo_hi = span.get(k)
        if lo_hi is None:
            print(f"  {k:<20}{n:>7}{'None':>11}{'':>11}", flush=True)
        else:
            print(f"  {k:<20}{n:>7}{lo_hi[0]:>11.4f}{lo_hi[1]:>11.4f}", flush=True)


def _loop(args, metrics, recorder, streams, pv, si, hub) -> None:
    """지표 계산·전송·시각화 루프. hub 경로와 hl2ss 경로가 같이 쓴다.

    pv 는 snapshot() 을, si 는 latest/trail 을 제공하면 된다. hub 경로에서는 HubSource
    하나가 셋을 다 겸한다.
    """
    print(f"지표 창 {args.window}s, 전송 {args.rate} Hz. q 또는 ESC 로 종료.", flush=True)
    period = 1.0 / args.rate
    next_tick = time.time()
    last_log = time.time()
    last_control = 0.0
    ticks = dropped = 0
    is_hub = isinstance(hub, HubSource)
    probe = next((s.audio_probe for s in streams if hasattr(s, "audio_probe")), None)
    span: dict[str, tuple[float, float]] = {}        # --diag: 블록 안 지표 최소/최대
    prev_n: dict[str, int] = {}                      # --diag: 수신율 계산용 직전 카운트
    prev_pv_n = 0                                    # 레이어가 PV 를 따라오는지 비교용

    try:
        while True:
            # '지금' 은 벽시계가 아니라 가장 최근 패킷의 시각이다. 패킷 timestamp 의 원점은
            # 앱 기동(hub) 또는 기기 부팅(hl2ss) 이라 epoch 와 축이 다르다. 벽시계를 쓰면
            # 모든 샘플이 창 밖으로 밀려 지표가 전부 NaN 이 된다.
            now_qpc = metrics.latest_timestamp
            if now_qpc is None:                      # 아직 한 프레임도 안 왔다
                # 스트림이 전부 open 에 실패했다면(IP 오타 등) 기다려도 오지 않는다.
                # HubSource 에는 error 속성이 없으므로 hub 경로는 여기 걸리지 않는다.
                if streams and all(getattr(s, "error", None) is not None for s in streams):
                    print("모든 스트림 연결 실패. 주소와 기기 앱 상태를 확인하라.", flush=True)
                    return
                time.sleep(period)
                continue
            values = metrics.current(now_qpc)
            ticks += 1

            if args.diag:
                for k, v in values.items():
                    if v is None or np.isnan(v):
                        continue
                    lo_hi = span.get(k)
                    span[k] = (v, v) if lo_hi is None else (min(lo_hi[0], v), max(lo_hi[1], v))

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
                comp = pv.snapshot_comp() if hasattr(pv, "snapshot_comp") else None
                if comp is not None:
                    cv2.imshow("Composite", comp)
                cv2.imshow("Metrics", render_metrics_panel(values))
                f = si.latest
                if f is not None and f.has_head:
                    cv2.imshow("Head", render_head_view(f.head_position, f.head_forward,
                                                        trail=list(si.trail)))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            # 기기는 제어를 일정 시간 못 받으면 스트림을 놓는다. 브로커가 조용히 사라져도
            # 기기가 마이크를 계속 잡고 있지 않도록 주기적으로 다시 보낸다.
            if is_hub and time.time() - last_control >= CONTROL_PERIOD_S:
                hub.send_control(send_audio=not args.no_audio, send_render=not args.no_render)
                last_control = time.time()

            if time.time() - last_log >= 5.0:
                elapsed = time.time() - last_log
                alive = ", ".join(f"{s.name}:{s.n}" for s in streams if s.running)
                drop = f"  hub_dropped={dropped}" if dropped else ""
                rs = next((s.render_stats for s in streams
                           if getattr(s, "want_render", False)), None)
                pv_rate = (pv.n - prev_pv_n) / elapsed if hasattr(pv, "n") else 0.0
                prev_pv_n = pv.n if hasattr(pv, "n") else 0
                rline, warn = rs.status(elapsed, pv_rate) if rs is not None else ("", None)
                print(f"  ticks={ticks}  {alive}{drop}"
                      + (f"  {rline}" if rline else ""), flush=True)
                if warn:
                    print(f"  ! {warn}", flush=True)
                    src = next((s for s in streams if getattr(s, "miss", None)), None)
                    if src is not None:
                        print(f"    {src.miss}", flush=True)
                if args.diag:
                    _print_diag(metrics, streams, probe, now_qpc, elapsed, span, prev_n)
                    prev_n = {s.name: s.n for s in streams}
                    span = {}
                if rs is not None:
                    rs.roll()
                last_log = time.time()

            next_tick += period
            time.sleep(max(0.0, next_tick - time.time()))
    except KeyboardInterrupt:
        print("\n중단", flush=True)
    finally:
        if is_hub:
            hub.send_control(send_audio=False, send_render=False)   # 기기가 즉시 놓게 한다
            time.sleep(0.2)
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
