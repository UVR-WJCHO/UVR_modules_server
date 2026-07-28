"""홀로렌즈 전송 데이터(HL2DATA) 녹화 → 파일(데이터셋).

브로커에서 HL2DATA 를 구독해 (상대시간, fid, protobuf bytes) 를 파일에 **스트리밍 저장**한다.
HL2DATA 안에는 이미지 + 머리/손/시선 + cam_to_world + intrinsics 가 모두 들어있으므로,
이 파일 하나로 전체 파이프라인을 오프라인 재생(play_hl2.py)할 수 있다.

온/오프:
  실행하면 녹화 ON. 콘솔에서 **Enter = 녹화 토글(ON/OFF)**, **q = 저장하고 종료**.
  프레임마다 파일에 바로 기록하므로 중간에 크래시 나도 그때까지는 보존됨.

실행:
  python record_hl2.py                       # rec.hl2rec, 브로커 기본
  python record_hl2.py walk1.hl2rec --host 143.248.96.81 --port 37001

포맷(한 레코드): <d I I> (상대초, fid_len, data_len) + fid + data
"""
import sys
import os
import time
import struct
import threading

import zmq

# protobuf 정의는 _comm/ 아래에 있음 (main_handtrack.py 의 modules/ 패턴과 동일)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_comm"))
try:
    import hl2_data_pb2 as _proto
except Exception as e:      # 녹화는 raw bytes 저장이라 pb2 없어도 동작
    _proto = None
    print(f"[rec] (경고) hl2_data_pb2 import 실패, protobuf 요약 생략: {e}")

SERVER_NAME = b"HL2_RECORDER"
RECV_KW = [b"HL2DATA"]

recording = True
quit_flag = False


def describe(data):
    """녹화 중인 protobuf 한 프레임이 유효한지/무엇을 담는지 확인용 요약."""
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
    out, host, port = "rec.hl2rec", "143.248.96.81", 37001
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--host":
            host = args[i + 1]; i += 1
        elif a == "--port":
            port = int(args[i + 1]); i += 1
        elif not a.startswith("--"):
            out = a
        i += 1
    return out, host, port


def stdin_loop():
    global recording, quit_flag
    for line in sys.stdin:
        c = line.strip().lower()
        if c == "q":
            quit_flag = True
            break
        recording = not recording          # Enter 또는 아무 글자 = 토글
        print(f"[rec] 녹화 {'ON' if recording else 'OFF'}")


def main():
    out, host, port = parse_args()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, SERVER_NAME)
    sock.setsockopt(zmq.RCVTIMEO, 500)     # quit_flag 확인용 타임아웃
    sock.connect(f"tcp://{host}:{port}")
    sock.send_multipart([b"", b"RECV_REG"] + RECV_KW + [SERVER_NAME, b"ALL"])

    threading.Thread(target=stdin_loop, daemon=True).start()
    print(f"[rec] {out} 녹화 시작(ON).  Enter=토글, q=저장종료")

    n = 0
    t0 = time.time()
    with open(out, "wb") as f:
        while not quit_flag:
            try:
                msg = sock.recv_multipart()
            except zmq.Again:
                continue
            if msg[1] == b"NOTIFY":
                _, _, kw, src, fid = msg
                sock.send_multipart([b"", b"DOWNLOAD", kw, src, fid])
            elif msg[1] == b"DATA_REPLY":
                _, _, kw, src, fid, data = msg
                if recording:
                    t = time.time() - t0
                    f.write(struct.pack("<dII", t, len(fid), len(data)))
                    f.write(fid)
                    f.write(data)
                    f.flush()
                    n += 1
                    if n == 1:
                        print(f"[rec] protobuf 확인: {describe(data)}")
                    if n % 30 == 0:
                        print(f"[rec] {n} frames")
    print(f"[rec] 저장 완료: {out} ({n} frames)")


if __name__ == "__main__":
    main()
