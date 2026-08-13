"""main_meshrecon 의 comm_hub 버전 — 캡처 -> 재구성 -> 조립 -> 합본 GLB.

기존 main_meshrecon.py 는 Hl2Manager(hl2ss 직접 스트리밍)로 RGB/depth 를 받았다.
이 버전은 수신을 comm_hub 로 바꾸고, 캡처를 modules_jointtrack 이 읽는 레이아웃으로
쌓아서 마지막에 정합·합본까지 한 번에 끝낸다.

  SPACE  유닛 캡처. 세그먼트된 물체를 TRELLIS 로 재구성해 part_<n>/ 에 넣는다.
         n 은 0 부터 증가한다.
  a      조립 캡처. 이미지·depth 만 저장한다. 직전 두 유닛을 이어 붙인 이름이
         붙는다 — 유닛이 0,1 이면 part_01, 0,1,2 면 part_12.
  ENTER  정합. 쌓인 캡처로 modules_jointtrack 을 돌려 유닛별 자세를 구하고,
         그 자세로 파츠를 하나의 GLB 로 합쳐 comm_hub 로 올린다
         (UPLOAD, kw=MESH_RESULT, payload=MeshResult). HTTP 서빙도 UDP 신호도
         쓰지 않는다 — HL2 는 손 결과와 같은 방식으로 구독해서 받아간다.
  q      종료.

조립 캡처는 mesh 가 없다. 그 프레임에 놓이는 메시는 Stage 1 이 만든 것이고,
조립체 자체의 재구성은 쓰이지 않는다 (meshalignment.frames 가 mesh 를 optional 로
읽는 이유).

유닛이 0,1,2 이고 조립이 01, 12 면 두 solve 가 유닛 1 을 공유하므로 하나의 사슬로
이어진다. 사슬이 끊기면 정합 단계에서 걸린다.

실행:
    conda activate uvr_integ
    python comm_hub.py --port 37001        # 터미널 1
    python main_meshrecon_comm.py          # 터미널 2
"""
import os
import sys

# modules/ 를 top-level 로 import 가능하게 (main_meshrecon.py 와 동일)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
# protobuf 정의는 _comm/ 아래
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_comm"))

import argparse
import datetime
import gc
import json
import shutil
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
import torch
import zmq
from PIL import Image

import hl2_data_pb2 as proto

from modules_console import (enable_cbreak_stdin, read_key_nonblocking,
                             restore_stdin_cbreak)


# --- CONFIGURATION ---
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 37001
RECV_KW = b"HL2DATA"
RESULT_KW = b"MESH_RESULT"
IDENTITY = b"MESHRECON"

flag_recon_mesh = True
flag_interactive_hotrack = True   # True: InteractiveHoTrackSegmentor, False: legacy HOSegmentor


# --- comm_hub 클라이언트 (최신 프레임만 유지) ---
class HubClient:
    """comm_hub DEALER 클라이언트. 수신은 백그라운드 스레드에서 최신 1프레임만 남긴다."""

    def __init__(self, host, port, recv_kw=RECV_KW, result_kw=RESULT_KW, identity=IDENTITY):
        self.ctx = zmq.Context()
        self.identity = identity
        self.result_kw = result_kw

        # 수신 소켓 (백그라운드 스레드 전용)
        self.rx = self.ctx.socket(zmq.DEALER)
        self.rx.setsockopt(zmq.IDENTITY, identity)
        self.rx.setsockopt(zmq.RCVTIMEO, 500)
        self.rx.connect(f"tcp://{host}:{port}")
        self.rx.send_multipart([b"", b"RECV_REG", recv_kw, identity, b"ALL"])

        # 송신 소켓 (메인 스레드 전용) — 같은 소켓을 두 스레드가 쓰지 않도록 분리
        self.tx = self.ctx.socket(zmq.DEALER)
        self.tx.setsockopt(zmq.IDENTITY, identity + b"_TX")
        self.tx.connect(f"tcp://{host}:{port}")

        self.q = Queue(maxsize=1)
        self._fid = 0
        self._stop = False
        threading.Thread(target=self._rx_loop, daemon=True).start()
        print(f"{identity.decode()} :: connected tcp://{host}:{port}, "
              f"recv={recv_kw.decode()} result={result_kw.decode()}", flush=True)

    def _rx_loop(self):
        while not self._stop:
            try:
                msg = self.rx.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            if msg[1] == b"NOTIFY":
                _, _, kw, src, fid = msg
                self.rx.send_multipart([b"", b"DOWNLOAD", kw, src, fid])
            elif msg[1] == b"DATA_REPLY":
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except Empty:
                        pass
                self.q.put(msg[5])

    def get_latest(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            return None

    def send_mesh(self, payload):
        """합본 GLB 를 UPLOAD. 프레임마다가 아니라 정합이 끝날 때 한 번 나간다."""
        self._fid += 1
        self.tx.send_multipart([b"", b"UPLOAD", self.result_kw, self.identity,
                                str(self._fid).encode(), payload])

    def close(self):
        self._stop = True
        self.rx.close()
        self.tx.close()
        self.ctx.term()


# --- 패킷 -> meshalignment 가 읽는 형태 ---
def decode_rgb(buf):
    if not buf:
        return None
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def decode_depth_m(buf):
    """RGB 정렬 16bit depth PNG(mm) -> float32 metre. meshalignment 는 m 로 읽는다."""
    if not buf:
        return None
    d = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[:, :, 0]
    return d.astype(np.float32) / 1000.0


def intrinsic_matrix(pkt, w, h):
    """패킷의 fx,fy,cx,cy 를 3x3 으로. 0 이면 화면 크기 기반 대략값."""
    fx, fy, cx, cy = pkt.fx, pkt.fy, pkt.cx, pkt.cy
    if fx <= 1 or fy <= 1:
        fx = fy = 0.8 * w
        cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def save_capture(dir_path: Path, color, depth_m, K, obj_mask):
    """meshalignment 가 요구하는 네 파일. mesh.glb 는 유닛 캡처에서만 따로 쓴다."""
    dir_path.mkdir(parents=True, exist_ok=True)
    masked = np.where(obj_mask[..., None] == 1, color, 3)
    cv2.imwrite(str(dir_path / "rgb.png"), color)
    cv2.imwrite(str(dir_path / "rgb_masked.png"), masked)
    np.save(dir_path / "depth.npy", depth_m)
    np.save(dir_path / "intrinsic.npy", K)


def fail(what: str, exc: BaseException):
    """실패를 눈에 띄게. 스택까지 남겨야 어디서 났는지 다시 물을 일이 없다."""
    line = "!" * 70
    print(f"\n{line}\n!!  FAILED: {what}\n!!  {type(exc).__name__}: {exc}\n{line}",
          flush=True)
    traceback.print_exc()
    print(f"{line}\n", flush=True)


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


# --- 정합 + 합본 ---
def assembly_label(units):
    """직전 두 유닛을 이어 붙인 이름. 유닛이 하나뿐이면 조립할 게 없다."""
    if len(units) < 2:
        return None
    return "".join(units[-2:])


def run_jointtrack(session_dir: Path, units, assemblies, device="cuda"):
    """쌓인 캡처로 Stage 1 + Stage 2 + 핸드오버까지. inputs/ 경로를 돌려준다.

    modules_jointtrack 의 CLI 를 그대로 함수로 부른다 — 같은 코드 경로라 CLI 로
    재현할 수 있다.
    """
    from meshalignment.assemble import (NotASimilarity, chain_solves,
                                        stage_meshes, write_transforms)
    from modules_jointtrack import fit_assembly, fit_parts

    seed_dir = session_dir / "stage1"
    mesh_dir = seed_dir / "meshes"

    print("=" * 70 + "\n[1/3] FIT EACH PART TO ITS OWN CAPTURE\n" + "=" * 70)
    fit_parts(session_dir, units, seed_dir, mesh_dir, device, girth_rounds=4)

    print("\n" + "=" * 70 + "\n[2/3] ASSEMBLE\n" + "=" * 70)
    solves = [fit_assembly(session_dir, mesh_dir, seed_dir, cid, members,
                           session_dir / f"C{cid}", device)
              for cid, members in assemblies]

    print("\n" + "=" * 70 + "\n[3/3] HAND OVER\n" + "=" * 70)
    merged = chain_solves(solves)
    missing = [u for u in units if u not in merged]
    if missing:
        raise RuntimeError(
            f"units never placed: {missing} — the assembly captures do not chain "
            f"through a shared unit")

    inputs = session_dir / "inputs"
    stage_meshes(mesh_dir, units, inputs / "glbs")
    try:
        parts = write_transforms(inputs / "transforms.json",
                                 [(u, merged[u]) for u in units])
    except NotASimilarity as exc:
        raise RuntimeError(f"pose is not a similarity: {exc}") from exc
    for i, (u, p) in enumerate(zip(units, parts)):
        print(f"  mesh_{i}.glb  <- unit {u:>4}   "
              f"t={[round(v, 4) for v in p['translation']]}  s={p['scale'][0]:.5f}")
    return inputs


def combine(inputs_dir: Path, n_units: int, out_glb: Path, out_metadata: Path):
    """정합된 파츠를 transforms.json 대로 하나의 GLB 로 합친다.

    metaobj_wrapper/combine_rocket_glb.py 를 그대로 쓴다 — 합본의 노드 계층은
    HL2 쪽이 기대하는 모양이라 여기서 다시 만들면 안 된다.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "metaobj_wrapper"))
    from combine_rocket_glb import (combine_parts, load_part_specs,
                                    specs_for_parts, write_metadata_json)

    part_files = [inputs_dir / "glbs" / f"mesh_{i}.glb" for i in range(n_units)]
    specs = specs_for_parts(load_part_specs(inputs_dir / "transforms.json"))
    combine_parts(part_files, specs, out_glb)
    write_metadata_json(out_metadata, specs)
    return out_glb


def build_mesh_result(glb_path: Path, metadata_path: Path, unit_names):
    """합본 GLB 를 MeshResult 로 직렬화. HL2 는 이걸 받아 바로 띄우면 된다."""
    r = proto.MeshResult()
    r.glb = glb_path.read_bytes()
    r.part_count = len(unit_names)
    r.part_names.extend(unit_names)
    r.metadata_json = metadata_path.read_text(encoding="utf-8")
    r.timestamp = time.monotonic()
    return r.SerializeToString()


# --- MAIN ---
def main():
    ap = argparse.ArgumentParser(description="HL2 meshrecon over comm_hub")
    ap.add_argument("--host", default=BROKER_HOST, help="comm_hub 브로커 IP")
    ap.add_argument("--port", type=int, default=BROKER_PORT)
    ap.add_argument("--session", default=None,
                    help="캡처를 쌓을 디렉토리 (기본: output/<timestamp>)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    session_dir = Path(args.session or
                       Path("output") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Session] {session_dir}")

    clear_gpu_memory()

    # 무거운 모델 import 는 여기서
    from modules_hotrack import build_interactive_hotrack_segmentor_from_env
    from modules_mesh import MeshReconstructor
    from modules_segment import HOSegmentor

    print("\n[Init] segmentor...")
    hosegmentor = (build_interactive_hotrack_segmentor_from_env()
                   if flag_interactive_hotrack else HOSegmentor())
    meshrecon = MeshReconstructor() if flag_recon_mesh else None
    if meshrecon is not None:
        print("[Init] MeshReconstructor ready")

    hub = HubClient(args.host, args.port)

    units = []           # 재구성이 끝난 유닛 이름, 캡처 순서
    assemblies = []      # (조립 캡처 이름, [그 안의 유닛])
    is_first_call = True
    frame_idx = 0

    print("\n[Ready] SPACE=유닛 캡처   a=조립 캡처   ENTER=정합+합본   q=종료")
    console_state = enable_cbreak_stdin()
    try:
        while True:
            data = hub.get_latest(timeout=1.0)
            if data is None:
                continue
            pkt = proto.HL2SensorPacket()
            pkt.ParseFromString(data)

            color = decode_rgb(pkt.image_data)
            depth_m = decode_depth_m(pkt.depth_data)
            if color is None or depth_m is None:
                continue
            H, W = color.shape[:2]
            if depth_m.shape[:2] != (H, W):
                depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)
            K = intrinsic_matrix(pkt, W, H)

            cv2.imshow("RGB", color)
            rgb_key = cv2.waitKey(1)
            if flag_interactive_hotrack:
                hosegmentor.handle_key(rgb_key)
                if hosegmentor.quit_requested:
                    break

            # 세그먼테이션
            if flag_interactive_hotrack:
                with torch.inference_mode():
                    _, result_masks, _ = hosegmentor.process_frame(color.copy())
            else:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    depth_cm = depth_m * 100.0
                    hotrack_datas = hosegmentor.get_hotrack_datas(color, depth_cm)
                    _, result_masks = hosegmentor.run(color.copy(), depth_cm, hotrack_datas,
                                                      is_first_call, frame_idx)
                    is_first_call = False
            frame_idx += 1
            if len(result_masks) == 0:
                continue

            obj_mask = result_masks[0]
            cv2.imshow("Segmentation", np.where(obj_mask[..., None] == 1, color, 3))
            cv2.waitKey(1)

            key = read_key_nonblocking(timeout=0.01)
            if not key:
                continue

            # ---------- 유닛 캡처 ----------
            if key == " ":
                name = str(len(units))
                cap = session_dir / f"part_{name}"
                save_capture(cap, color, depth_m, K, obj_mask)
                print(f"\n[Unit {name}] captured -> {cap}")

                if meshrecon is None:
                    units.append(name)
                    continue
                try:
                    masked_rgba = np.dstack([color, (obj_mask.astype(np.uint8) * 255)])
                    mesh_glb, preview = meshrecon.run(
                        Image.fromarray(masked_rgba, "RGBA"), return_preview=True)
                    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Mesh preview", preview_bgr)
                    cv2.waitKey(1)
                    cv2.imwrite(str(cap / "mesh_preview.png"), preview_bgr)
                    mesh_glb.export(cap / "mesh.glb")
                    units.append(name)
                    print(f"[Unit {name}] mesh.glb written. units={units}")
                except Exception as e:
                    shutil.rmtree(cap, ignore_errors=True)
                    fail(f"unit {name} reconstruction — capture discarded, "
                         f"units still {units}", e)
                finally:
                    clear_gpu_memory()

            # ---------- 조립 캡처 ----------
            elif key == "a":
                name = assembly_label(units)
                if name is None:
                    print("[Assembly] need two reconstructed units first")
                    continue
                members = units[-2:]
                cap = session_dir / f"part_{name}"
                save_capture(cap, color, depth_m, K, obj_mask)
                assemblies = [(c, m) for c, m in assemblies if c != name]
                assemblies.append((name, members))
                print(f"\n[Assembly {name}] captured (units {members}) -> {cap}")

            # ---------- 정합 + 합본 ----------
            elif key in ("\r", "\n"):
                if not assemblies:
                    print("[Align] no assembly capture yet")
                    continue
                print(f"\n[Align] units={units}  assemblies={[c for c, _ in assemblies]}")
                if meshrecon is not None:
                    meshrecon.pipeline.cpu()   # 정합이 쓸 VRAM 확보
                    clear_gpu_memory()
                try:
                    inputs = run_jointtrack(session_dir, units, assemblies, args.device)
                    glb = session_dir / "combined.glb"
                    meta = session_dir / "combined_metadata.json"
                    combine(inputs, len(units), glb, meta)
                    payload = build_mesh_result(glb, meta, units)
                    hub.send_mesh(payload)
                    print(f"\n[Align] combined {len(units)} units -> {glb} "
                          f"({glb.stat().st_size / 1024:.0f} KB)")
                    print(f"[Send] UPLOAD {RESULT_KW.decode()} "
                          f"({len(payload) / 1024:.0f} KB) -> comm_hub")
                except Exception as e:
                    fail(f"alignment (units={units}, "
                         f"assemblies={[c for c, _ in assemblies]})", e)
                finally:
                    if meshrecon is not None:
                        meshrecon.pipeline.cuda()
                    clear_gpu_memory()
                print("\n[Ready] SPACE=유닛   a=조립   ENTER=정합+합본   q=종료")

            elif key == "q":
                break

    except KeyboardInterrupt:
        print("\n[Info] shutting down...")
    finally:
        (session_dir / "session.json").write_text(json.dumps(
            {"units": units, "assemblies": [{"cid": c, "units": m} for c, m in assemblies]},
            indent=2))
        if hasattr(hosegmentor, "close"):
            hosegmentor.close()
        del hosegmentor
        if meshrecon is not None:
            del meshrecon
        clear_gpu_memory()
        hub.close()
        cv2.destroyAllWindows()
        restore_stdin_cbreak(console_state)


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
