"""녹화된 HL2DATA(.hl2rec) 재생 — 홀로렌즈 없이 브로커로 전송한다.

record_hl2.py 로 저장한 파일을 읽어 `UPLOAD HL2DATA` 로 전송(원래 프레임 타이밍 보존).
fid 는 재생 시 단조 증가로 새로 매기므로 --loop 반복해도 충돌 없음.
기기 식별자(--id)는 HL2Client 의 clientIdentity 와 동일하게 두면 홀로렌즈가 보낸 것처럼 동작.

실행:
  python play_hl2.py rec.hl2rec
  python play_hl2.py walk1.hl2rec --host 143.248.96.81 --port 37001 --loop --speed 1.0 --id HoloLens2_User
"""
import sys
import os
import time
import struct

import zmq

# protobuf 정의는 _comm/ 아래에 있음 (main_handtrack.py 의 modules/ 패턴과 동일)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_comm"))
try:
    import hl2_data_pb2 as _proto
except Exception as e:      # 재생은 raw bytes 전송이라 pb2 없어도 동작
    _proto = None
    print(f"[play] (경고) hl2_data_pb2 import 실패, protobuf 요약 생략: {e}")


def describe(data):
    """재생할 protobuf 한 프레임 요약 — 파일이 유효한 HL2DATA 인지 확인용."""
    if _proto is None:
        return "(pb2 없음)"
    try:
        p = _proto.HL2SensorPacket()
        p.ParseFromString(data)
    except Exception as e:
        return f"(파싱 실패: {e})"
    depth = f"on {p.depth_width}x{p.depth_height} {len(p.depth_data)}B" if (p.depth_enabled or p.depth_data) else "off"
    return (f"img={len(p.image_data)}B  depth={depth}  "
            f"hands R/L={len(p.right_hand)}/{len(p.left_hand)}  "
            f"intr(fx,fy)=({p.fx:.1f},{p.fy:.1f})  pts={len(p.depth_points)//3}")


def parse_args():
    path, host, port = None, "143.248.96.81", 37001
    loop, speed, ident = False, 1.0, "HoloLens2_User"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--host":
            host = args[i + 1]; i += 1
        elif a == "--port":
            port = int(args[i + 1]); i += 1
        elif a == "--loop":
            loop = True
        elif a == "--speed":
            speed = float(args[i + 1]); i += 1
        elif a == "--id":
            ident = args[i + 1]; i += 1
        elif not a.startswith("--"):
            path = a
        i += 1
    return path, host, port, loop, speed, ident


def records(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(16)
            if len(head) < 16:
                break
            t, fl, dl = struct.unpack("<dII", head)
            f.read(fl)                    # fid (재생 시 새로 매기므로 버림)
            data = f.read(dl)
            if len(data) < dl:
                break
            yield t, data


def main():
    path, host, port, loop, speed, ident = parse_args()
    if not path:
        print("사용법: python play_hl2.py rec.hl2rec "
              "[--host IP --port 37001 --loop --speed 1.0 --id NAME]")
        return
    if not os.path.exists(path):
        print(f"[play] 파일 없음: {path}")
        return

    # protobuf 확인: 첫 프레임 요약 + 총 프레임 수
    total = 0
    first = None
    for _t, d in records(path):
        if first is None:
            first = d
        total += 1
    if total == 0:
        print(f"[play] 빈 파일이거나 포맷 불일치: {path}")
        return
    print(f"[play] protobuf 확인: {total} frames, frame0 -> {describe(first)}")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, ident.encode())
    sock.connect(f"tcp://{host}:{port}")
    print(f"[play] {path} -> {host}:{port}  id={ident}  loop={loop}  speed={speed}")

    counter = 0
    try:
        while True:
            t_prev = None
            for t, data in records(path):
                if t_prev is not None:
                    dt = (t - t_prev) / speed
                    if dt > 0:
                        time.sleep(dt)
                t_prev = t
                fid = str(counter).encode(); counter += 1
                sock.send_multipart([b"", b"UPLOAD", b"HL2DATA", ident.encode(), fid, data])
                if counter % 30 == 0:
                    print(f"[play] sent {counter}")
            if not loop:
                break
    except KeyboardInterrupt:
        print("\n[play] 중단")
    print(f"[play] 전송 완료: {counter} frames")


if __name__ == "__main__":
    main()
