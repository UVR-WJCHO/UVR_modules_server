"""Audio load: 주변 사운드 세기 + 오디오 밀도(Spectral Flux).

- audio_density : Spectral Flux — 프레임 간 스펙트럼의 양의 변화량 누적.
                  소리가 얼마나 활발히 변하는지(발화·활동 vs 정적)를 나타냄.
- audio_level  : 프레임 RMS — 주변 사운드 세기.

estimate_audio_density()는 순수 함수(이전 스펙트럼을 인자로 받아 새 스펙트럼을 반환)라
라이브 오디오 루프(AudioThread)와 오프라인 배치(compute)에서 동일하게 재사용한다.
"""

import os
import wave
import numpy as np

from . import common as C


def estimate_audio_density(frame, prev_spectrum, window_size=1024, scaling_factor=1.0):
    """Spectral Flux 기반 오디오 밀도.

    frame: 1D 샘플 배열. prev_spectrum: 이전 프레임의 크기 스펙트럼(없으면 None).
    반환 (density, current_spectrum). density = Σ max(0, |cur| − |prev|) × scaling_factor.
    상태(previousSpectrum)는 호출자가 반환된 current_spectrum을 넘겨받아 관리한다.
    """
    x = np.asarray(frame)
    if np.issubdtype(x.dtype, np.integer):
        # 기기에서 오는 PCM 은 int16 이다. hl2ss 가 AAC 를 디코드해 주던 float(+-1.0) 과
        # 스케일이 32768 배 달라, 그대로 FFT 하면 density 가 표시 상한에 바로 붙는다.
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0)           # 다채널 -> 모노
    n = window_size
    x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
    mag = np.abs(np.fft.rfft(x * np.hanning(n)))
    if prev_spectrum is None:
        return 0.0, mag
    density = float(np.sum(np.maximum(0.0, mag - prev_spectrum))) * scaling_factor
    return density, mag


def _load_wav_mono(path):
    w = wave.open(path, 'rb')
    sr, ch = w.getframerate(), w.getnchannels()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def _audio_start_qpc(session, fallback):
    """audio_metadata.txt의 start_timestamp_qpc. 없으면 fallback(세션 시작, 근사)."""
    p = os.path.join(session, 'audio', 'audio_metadata.txt')
    if os.path.isfile(p):
        for line in open(p):
            if line.startswith('start_timestamp_qpc:'):
                v = line.split(':', 1)[1].strip()
                if v not in ('None', ''):
                    return int(v)
    return fallback


def compute(session, out_ts, W, window_size=1024, scaling_factor=1.0):
    path = os.path.join(session, 'audio', 'audio.wav')
    if not os.path.isfile(path):
        return {}
    x, sr = _load_wav_mono(path)
    t0 = _audio_start_qpc(session, int(out_ts[0]))
    n = window_size
    n_frames = 1 + (len(x) - n) // n if len(x) >= n else 0
    if n_frames <= 0:
        return {}

    dens = np.empty(n_frames)
    rms = np.empty(n_frames)
    ts = np.empty(n_frames, np.int64)
    prev = None
    for i in range(n_frames):
        fr = x[i * n:i * n + n]
        dens[i], prev = estimate_audio_density(fr, prev, n, scaling_factor)
        rms[i] = np.sqrt(np.mean(fr * fr))
        ts[i] = t0 + int((i * n + n / 2) / sr * C.QPC)

    snr, noise_floor = _windowed_snr(ts, rms, out_ts, W)
    return {
        'audio_density': C.windowed(ts, dens, out_ts, W, 'mean'),
        'audio_level': C.windowed(ts, rms, out_ts, W, 'rms'),
        'audio_snr': snr,                  # peak-to-floor 동적범위 (dB)
        'audio_noise_floor': noise_floor,  # 배경소음 바닥 레벨 (dBFS)
    }


def _windowed_snr(fts, rms, out_ts, W, p_lo=10, p_hi=90):
    """윈도우별 SNR(=20·log10(P90/P10) dB)과 noise_floor(=20·log10(P10) dBFS).

    신호/소음 분리 없이 프레임 RMS의 peak-to-floor로 근사. (해석은 후처리)
    """
    K = len(out_ts); Wt = int(W * C.QPC)
    snr = np.full(K, np.nan); nf = np.full(K, np.nan)
    for k, t in enumerate(out_ts):
        a = np.searchsorted(fts, t - Wt)
        b = np.searchsorted(fts, t, 'right')
        seg = rms[a:b]
        if seg.size >= 5:
            lo = np.percentile(seg, p_lo) + 1e-9
            hi = np.percentile(seg, p_hi) + 1e-9
            nf[k] = float(20 * np.log10(lo))
            snr[k] = float(20 * np.log10(hi / lo))
    return snr, nf
