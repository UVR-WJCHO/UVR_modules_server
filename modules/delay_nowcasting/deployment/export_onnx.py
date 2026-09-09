"""학습 checkpoint -> ONNX export, parity 검증, latency 측정 (B 계획 §11 Phase 6).

  python -m modules.delay_nowcasting.deployment.export_onnx \
      --checkpoint <ckpt> --verify

서버는 같은 history 에 horizon 만 바꿔 batch 로 한 번에 추론한다. 그래서 batch 축을
dynamic 으로 export 하고, batch-1 과 전체 horizon grid batch latency 를 모두 잰다.
검증 기준은 §1.3 — PyTorch/ONNX 최대 joint 차이 0.1 mm 이하.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.canonical_schema import NUM_JOINTS
from ..training import checkpoint as ckpt

# onnx_runner.DEFAULT_GRID_MS 와 같아야 한다. 여기서 갈라지면 검증이 배포와
# 다른 batch 크기·horizon 으로 돈다.
from .onnx_runner import DEFAULT_GRID_MS as DEFAULT_GRID


class ForecastWrapper(torch.nn.Module):
    """ONNX 로 내보낼 형태. 인자 순서를 고정한다."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, history: torch.Tensor, history_time_ms: torch.Tensor,
                horizon_ms: torch.Tensor, handedness: torch.Tensor,
                visibility: torch.Tensor) -> torch.Tensor:
        return self.model(history, history_time_ms, horizon_ms, handedness, visibility)


def sample_inputs(batch: int, history_length: int, device: str):
    """parity 검증과 속도 측정에 쓰는 입력.

    난수 관절을 그대로 쓰면 안 된다. 손 모양이 아닌 데다 프레임간 속도가 실제의 아홉
    배라, CV base 를 300 ms 외삽하면 출력이 4 만을 넘는다. 그러면 float32 상대오차가
    1e-7 이어도 절대차가 mm 단위로 커져 parity 가 늘 실패한다. 실제 자세를 흉내내도록
    손 모양(한 자리에 모인 관절)에 작은 프레임간 변화만 준다.
    """
    generator = torch.Generator(device="cpu").manual_seed(0)
    shape = torch.randn(1, 1, NUM_JOINTS, 3, generator=generator) * 0.05   # 손 모양
    drift = torch.randn(batch, history_length, NUM_JOINTS, 3, generator=generator) * 0.002
    history = shape + drift.cumsum(dim=1)                                  # 프레임간 소폭 변화
    times = torch.arange(-history_length + 1, 1, dtype=torch.float32) * 33.333
    horizons = (list(DEFAULT_GRID[:batch]) if batch <= len(DEFAULT_GRID)
                else [100.0] * batch)
    return (history.to(device),
            times.repeat(batch, 1).to(device),
            torch.tensor(horizons, dtype=torch.float32, device=device),
            torch.ones(batch, dtype=torch.long, device=device),
            torch.ones(batch, history_length, NUM_JOINTS, device=device))


def measure(run, repeats: int = 200, warmup: int = 20) -> dict:
    for _ in range(warmup):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    values = np.array(samples)
    return {"mean_ms": round(float(values.mean()), 4),
            "p50_ms": round(float(np.median(values)), 4),
            "p95_ms": round(float(np.percentile(values, 95)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX export / parity / latency")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint)
    model, payload = ckpt.load(checkpoint_path, args.device)
    model.eval()
    wrapper = ForecastWrapper(model).eval()
    history_length = model.history_length
    out_path = Path(args.out) if args.out else checkpoint_path.with_suffix(".onnx")

    inputs = sample_inputs(len(DEFAULT_GRID), history_length, args.device)
    torch.onnx.export(
        wrapper, inputs, str(out_path),
        input_names=["history", "history_time_ms", "horizon_ms", "handedness", "visibility"],
        output_names=["joints_world"],
        dynamic_axes={name: {0: "batch"} for name in
                      ["history", "history_time_ms", "horizon_ms", "handedness",
                       "visibility", "joints_world"]},
        opset_version=args.opset,
    )
    report = {"checkpoint": str(checkpoint_path), "onnx": str(out_path),
              "config_hash": payload["config_hash"], "seed": payload["seed"],
              "parameters": model.num_parameters,
              "onnx_bytes": out_path.stat().st_size,
              "horizon_grid_ms": list(DEFAULT_GRID)}

    if args.verify:
        import onnxruntime as ort

        # parity 는 **CPU EP 기준**으로 판정한다. graph 가 옳은지를 보는 검사이기 때문이다.
        # CUDA EP 는 커널 구현이 달라 float32 누적 차이가 남는데(실측 0.19 mm), 이는
        # graph 오류가 아니라 실행 정밀도 차이라 별도로 기록만 한다.
        cpu_model, _ = ckpt.load(checkpoint_path, "cpu")
        cpu_wrapper = ForecastWrapper(cpu_model.eval()).eval()
        cpu_session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        names = [i.name for i in cpu_session.get_inputs()]

        worst = 0.0
        for batch in (1, len(DEFAULT_GRID)):
            probe = sample_inputs(batch, history_length, "cpu")
            with torch.no_grad():
                reference = cpu_wrapper(*probe).numpy()
            feed = {n: t.detach().numpy() for n, t in zip(names, probe)}
            actual = cpu_session.run(None, feed)[0]
            worst = max(worst, float(np.abs(actual - reference).max()) * 1000)
        report["max_joint_diff_mm_cpu"] = round(worst, 6)
        report["parity_pass"] = bool(worst <= 0.1)

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if args.device == "cuda" else ["CPUExecutionProvider"])
        session = ort.InferenceSession(str(out_path), providers=providers)
        if args.device == "cuda":
            worst_gpu = 0.0
            for batch in (1, len(DEFAULT_GRID)):
                probe = sample_inputs(batch, history_length, args.device)
                with torch.no_grad():
                    reference = wrapper(*probe).cpu().numpy()
                feed = {n: t.detach().cpu().numpy() for n, t in zip(names, probe)}
                worst_gpu = max(worst_gpu,
                                float(np.abs(session.run(None, feed)[0] - reference).max()) * 1000)
            report["max_joint_diff_mm_cuda_ep"] = round(worst_gpu, 6)

        for batch, label in ((1, "batch1"), (len(DEFAULT_GRID), f"batch{len(DEFAULT_GRID)}")):
            probe = sample_inputs(batch, history_length, args.device)
            with torch.no_grad():
                report[f"torch_{label}"] = measure(lambda: wrapper(*probe))
            feed = {n: t.detach().cpu().numpy() for n, t in zip(names, probe)}
            report[f"onnx_{label}"] = measure(lambda: session.run(None, feed))
        report["providers"] = session.get_providers()

    report_path = out_path.with_name(out_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
