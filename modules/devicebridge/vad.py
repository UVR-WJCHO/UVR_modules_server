"""webrtcvad 래퍼. 20 ms PCM 청크 하나에 대해 발화 여부를 돌려준다.

와치 앱의 AudioStreamer 스펙에 맞춘다: 16 kHz, int16, mono, 20 ms = 640 bytes.
청크 길이가 어긋나면 webrtcvad 가 예외를 내므로 여기서 잘라내거나 0 으로 채운다.

temp/Test/vad_processor.py 를 옮겨온 것이다. base64 디코드는 호출자가 한다.
"""
from __future__ import annotations

import webrtcvad


class VadProcessor:
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16_000, frame_ms: int = 20):
        if aggressiveness not in range(4):
            raise ValueError("aggressiveness must be 0-3")
        if sample_rate not in (8_000, 16_000, 32_000):
            raise ValueError("sample_rate must be 8000, 16000, or 32000")
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30")
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, pcm: bytes) -> bool:
        n = self.frame_bytes
        if len(pcm) < n:
            pcm = pcm + b"\x00" * (n - len(pcm))
        elif len(pcm) > n:
            pcm = pcm[:n]
        return self._vad.is_speech(pcm, self.sample_rate)
