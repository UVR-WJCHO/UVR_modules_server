"""세션 녹화. 이미지·이벤트는 스트리밍 append, 고정폭 배열은 NPZ 로 주기 flush.

원본(Adaptive-MM-UI)은 세션이 끝날 때 SI 를 한 번에 NPZ 로 저장했다. 실험 중 크래시가
나면 그 세션이 통째로 사라진다. 여기서는 배열도 주기적으로 flush 해서 유실을 마지막
몇 초로 한정한다.

    output/adaptivemm/<session>/
      meta.json          세션 설정과 스트림 목록
      pv/<qpc>.jpg       PV 프레임 (도착 즉시 기록)
      metrics.jsonl      실시간 지표 한 줄에 한 tick (append)
      si.npz             head/eye/hand 배열 (주기 flush)
      imu_accel.npz      IMU 가속도계 (주기 flush)
      imu_gyro.npz       IMU 자이로
      audio.wav          마이크 (스트리밍 write)

시각 단위는 HoloLens QPC(100 ns) 정수다. 초로 보려면 1e7 로 나눈다.
"""
from __future__ import annotations

import json
import os
import threading
import time
import wave

import cv2
import numpy as np

FLUSH_INTERVAL_S = 10.0     # 배열을 디스크로 내리는 주기


class _ArrayLog:
    """같은 키 집합의 프레임을 쌓다가 주기적으로 NPZ 로 덮어쓴다.

    덮어쓰기라 파일이 항상 '지금까지 전부'를 담는다. append 형식이 아니므로 읽는 쪽이
    단순하고, 중간에 죽어도 마지막 flush 시점까지는 남는다.
    """

    def __init__(self, path: str):
        self.path = path
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()

    def append(self, row: dict) -> None:
        with self._lock:
            self._rows.append(row)
            due = time.time() - self._last_flush >= FLUSH_INTERVAL_S
        if due:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            rows = list(self._rows)
            self._last_flush = time.time()
        if not rows:
            return
        cols = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        tmp = self.path + ".tmp"
        # 경로 문자열을 주면 numpy 가 .npz 를 덧붙인다. 파일 객체로 넘겨 이름을 고정한다.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **cols)
        os.replace(tmp, self.path)      # 부분 쓰기 상태가 보이지 않게

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)


class SessionRecorder:
    """한 세션의 모든 출력을 관리한다. 스레드에서 동시에 불러도 된다."""

    def __init__(self, root: str, session: str, meta: dict | None = None):
        self.dir = os.path.join(root, session)
        self.pv_dir = os.path.join(self.dir, "pv")
        os.makedirs(self.pv_dir, exist_ok=True)

        self.session = session
        self.n_pv = 0
        self._t0 = time.time()

        self._metrics_f = open(os.path.join(self.dir, "metrics.jsonl"), "a", buffering=1)
        self._metrics_lock = threading.Lock()

        self.si = _ArrayLog(os.path.join(self.dir, "si.npz"))
        self.accel = _ArrayLog(os.path.join(self.dir, "imu_accel.npz"))
        self.gyro = _ArrayLog(os.path.join(self.dir, "imu_gyro.npz"))

        self._wav = None
        self._wav_lock = threading.Lock()

        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump({"session": session,
                       "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "timestamp_unit": "HoloLens QPC (100ns)",
                       **(meta or {})}, f, indent=2, ensure_ascii=False)

    # --- PV ---------------------------------------------------------------
    def write_pv(self, timestamp: int, bgr: np.ndarray) -> None:
        cv2.imwrite(os.path.join(self.pv_dir, f"{timestamp}.jpg"), bgr)
        self.n_pv += 1

    # --- 실시간 지표 -------------------------------------------------------
    def write_metrics(self, timestamp: int, values: dict) -> None:
        row = {"timestamp": int(timestamp)}
        for k, v in values.items():
            row[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        line = json.dumps(row, ensure_ascii=False)
        with self._metrics_lock:
            self._metrics_f.write(line + "\n")

    # --- Spatial Input -----------------------------------------------------
    def write_si(self, frame) -> None:
        """si_adapter.SIFrame 하나. 유효하지 않은 항목은 NaN 으로 채워 폭을 맞춘다."""
        nan3 = np.full(3, np.nan, np.float32)
        z = lambda h, k, shape, dt: (h[k] if h is not None
                                     else np.full(shape, np.nan if dt == np.float32 else 0, dt))
        L, R = frame.hand_left, frame.hand_right
        self.si.append({
            "timestamp": np.int64(frame.timestamp),
            "head_valid": np.bool_(frame.has_head),
            "eye_valid": np.bool_(frame.has_eye),
            "hand_left_valid": np.bool_(L is not None),
            "hand_right_valid": np.bool_(R is not None),
            "head_position": frame.head_position if frame.has_head else nan3,
            "head_forward": frame.head_forward if frame.has_head else nan3,
            "head_up": frame.head_up if frame.has_head else nan3,
            "eye_origin": frame.eye_origin if frame.has_eye else nan3,
            "eye_direction": frame.eye_direction if frame.has_eye else nan3,
            "hand_left_position": z(L, "position", (26, 3), np.float32),
            "hand_left_orientation": z(L, "orientation", (26, 4), np.float32),
            "hand_right_position": z(R, "position", (26, 3), np.float32),
            "hand_right_orientation": z(R, "orientation", (26, 4), np.float32),
        })

    # --- IMU ---------------------------------------------------------------
    def write_imu(self, kind: str, timestamp: int, xyz: np.ndarray) -> None:
        """kind: 'accel' | 'gyro'. xyz 는 (N, 3) 배치."""
        log = self.accel if kind == "accel" else self.gyro
        log.append({"timestamp": np.int64(timestamp),
                    "xyz": np.asarray(xyz, np.float32)})

    # --- Audio -------------------------------------------------------------
    def open_audio(self, channels: int, sample_rate: int, sampwidth: int = 4) -> None:
        with self._wav_lock:
            self._wav = wave.open(os.path.join(self.dir, "audio.wav"), "wb")
            self._wav.setnchannels(channels)
            self._wav.setsampwidth(sampwidth)
            self._wav.setframerate(sample_rate)

    def write_audio(self, samples: np.ndarray) -> None:
        with self._wav_lock:
            if self._wav is not None:
                self._wav.writeframes(np.asarray(samples, np.float32).tobytes())

    # --- 종료 --------------------------------------------------------------
    def close(self) -> dict:
        for log in (self.si, self.accel, self.gyro):
            log.flush()
        with self._metrics_lock:
            self._metrics_f.close()
        with self._wav_lock:
            if self._wav is not None:
                self._wav.close()
        stats = {"session": self.session, "duration_s": round(time.time() - self._t0, 1),
                 "pv_frames": self.n_pv, "si_frames": len(self.si),
                 "imu_accel_packets": len(self.accel), "imu_gyro_packets": len(self.gyro)}
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(stats, f, indent=2)
        return stats
