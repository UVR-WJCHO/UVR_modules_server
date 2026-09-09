"""서버가 쓰는 ONNX forecast 추론 (B 계획 §5, §11 Phase 6).

같은 history 에 horizon 만 바꿔 한 번에 batch 로 돌린다. 모델이 horizon 을 입력으로
받으므로 forecast grid 전체가 batch 하나다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.canonical_schema import NUM_JOINTS

DEFAULT_GRID_MS = (0.0, 33.0, 66.0, 100.0, 133.0, 166.0, 199.0, 233.0, 266.0, 300.0)


class ForecastRunner:
    def __init__(self, onnx_path: str | Path, grid_ms=DEFAULT_GRID_MS,
                 providers: list[str] | None = None):
        import onnxruntime as ort

        self.grid_ms = np.asarray(grid_ms, dtype=np.float32)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(onnx_path), options,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.model_version = Path(onnx_path).stem

    def run(self, history_joints: np.ndarray, history_time_ms: np.ndarray,
            handedness: str, visibility: np.ndarray | None = None) -> np.ndarray:
        """(N, 21, 3) history -> (H, 21, 3) forecast grid."""
        batch = len(self.grid_ms)
        history = np.repeat(np.asarray(history_joints, np.float32)[None], batch, axis=0)
        times = np.repeat(np.asarray(history_time_ms, np.float32)[None], batch, axis=0)
        if visibility is None:
            visibility = np.ones(history.shape[:3], np.float32)
        else:
            visibility = np.repeat(np.asarray(visibility, np.float32)[None], batch, axis=0)
        feed = dict(zip(self.input_names, [
            history, times, self.grid_ms,
            np.full(batch, 1 if handedness == "RIGHT" else 0, np.int64),
            visibility,
        ]))
        return self.session.run(None, feed)[0].reshape(batch, NUM_JOINTS, 3)
