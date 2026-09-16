"""comm_hub 에 붙는 DEALER 클라이언트. 모든 진입점이 이 하나를 쓴다.

이 파일이 생기기 전에는 `main_handtrack.py`, `main_handtrack_forecast.py`,
`main_meshrecon.py` 가 같은 클래스를 각자 65~70줄씩 들고 있었고, 뷰어는 zmq 를
직접 열었다. 규약이 하나인데 구현이 넷이라 이미 서로 달라지기 시작했으므로
(프레임 언패킹 방식, 스레드 시작 방식, 결과 전송 메서드 이름) 여기로 모은다.

프로토콜은 `comm_hub.py` 가 기준이다:

    RECV_REG   [b"", b"RECV_REG",   KW, SOURCE, TARGET]        구독 등록
    NOTIFY     [b"", b"NOTIFY",     KW, SOURCE, FID]           도착 알림 (브로커 -> 구독자)
    DOWNLOAD   [b"", b"DOWNLOAD",   KW, SOURCE, FID]           본문 요청
    DATA_REPLY [b"", b"DATA_REPLY", KW, SOURCE, FID, DATA]     본문 응답
    UPLOAD     [b"", b"UPLOAD",     KW, SOURCE, FID, DATA]     결과 업로드

수신은 백그라운드 스레드가 돌며 최신 한 프레임만 남긴다(conflate). 처리가 느리면
오래된 프레임은 버린다 — 지연된 자세를 뒤늦게 그리는 것보다 건너뛰는 편이 낫다.
송신은 별도 소켓을 쓴다. 같은 소켓을 두 스레드가 만지지 않게 하기 위해서다.
"""
from __future__ import annotations

import threading
from queue import Empty, Queue

import zmq

# --- 브로커 기본값 -------------------------------------------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 37001

# --- keyword. 채널 이름이자 브로커의 캐시 단위다 -------------------------------
KW_HL2DATA = b"HL2DATA"              # HL2 -> 서버. RGB/depth 센서 패킷
KW_SERVER_RESULT = b"SERVER_RESULT"  # 서버 -> HL2. 단일 자세 결과
KW_HAND_FORECAST = b"HAND_FORECAST"  # 서버 -> HL2. horizon grid 결과
KW_MESH_RESULT = b"MESH_RESULT"      # 서버 -> HL2. 정합된 합본 GLB
KW_USER_STATE = b"USER_STATE"        # 서버 -> 구독자. 정량화된 사용자 상태 지표 (JSON)
KW_HL2_CONTROL = b"HL2_CONTROL"      # 서버 -> HL2. 온디맨드 스트림 on/off
KW_HL2_AUDIO = b"HL2_AUDIO"          # HL2 -> 서버. 마이크 청크
KW_HL2_RENDER = b"HL2_RENDER"        # HL2 -> 서버. AR 레이어 (홀로그램만, 배경 투명)
KW_HL2_IMU = b"HL2_IMU"              # HL2 -> 서버. (현재 미사용)

RECV_TIMEOUT_MS = 500                # rx 소켓 타임아웃. 종료 신호를 확인할 주기


class HubClient:
    """구독 한 채널, 업로드 한 채널.

    Args:
        host, port:  comm_hub 주소
        recv_kw:     구독할 keyword
        result_kw:   업로드할 keyword. None 이면 송신 소켓을 열지 않는다(읽기 전용)
        identity:    DEALER identity. 진입점마다 달라야 한다. comm_hub 는 source 단위로
                     구독을 덮어쓰므로, 한 프로세스가 여러 keyword 를 구독하려면 클라이언트
                     마다 identity 가 달라야 한다
        queue_size:  1 이면 최신 한 프레임만 남긴다(conflate). 프레임을 건너뛰어도 되는
                     영상에는 이쪽이 맞다. 오디오처럼 연속성이 필요한 스트림은 더 크게 잡아
                     중간 청크가 버려지지 않게 한다
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 recv_kw: bytes = KW_HL2DATA, result_kw: bytes | None = None,
                 identity: bytes = b"CLIENT", queue_size: int = 1):
        self.ctx = zmq.Context()
        self.identity = identity
        self.result_kw = result_kw

        self.rx = self.ctx.socket(zmq.DEALER)
        self.rx.setsockopt(zmq.IDENTITY, identity)
        self.rx.setsockopt(zmq.RCVTIMEO, RECV_TIMEOUT_MS)
        # 브로커가 죽었거나 처음부터 없으면 보내지 못한 메시지가 큐에 남는다. LINGER 가
        # 기본값(무한)이면 ctx.term() 이 그것들을 기다리며 종료가 걸린다. 종료 시점에
        # 전달하지 못한 것은 버린다.
        self.rx.setsockopt(zmq.LINGER, 0)
        self.rx.connect(f"tcp://{host}:{port}")
        self.rx.send_multipart([b"", b"RECV_REG", recv_kw, identity, b"ALL"])

        self.tx = None
        if result_kw is not None:
            self.tx = self.ctx.socket(zmq.DEALER)
            self.tx.setsockopt(zmq.IDENTITY, identity + b"_TX")
            self.tx.setsockopt(zmq.LINGER, 0)
            self.tx.connect(f"tcp://{host}:{port}")

        self.q: Queue = Queue(maxsize=max(1, queue_size))
        self.n_arrived = 0      # 브로커에서 실제로 도착한 프레임
        self.n_dropped = 0      # 루프가 처리 중이라 버린 프레임
        self._fid = 0
        self._stop = False
        threading.Thread(target=self._rx_loop, daemon=True).start()

        sent = result_kw.decode() if result_kw else "-"
        print(f"{identity.decode()} :: connected tcp://{host}:{port}, "
              f"recv={recv_kw.decode()} result={sent}", flush=True)

    def _rx_loop(self) -> None:
        while not self._stop:
            try:
                msg = self.rx.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break                       # 소켓/컨텍스트 종료
            if msg[1] == b"NOTIFY":
                _, _, kw, src, fid = msg
                self.rx.send_multipart([b"", b"DOWNLOAD", kw, src, fid])
            elif msg[1] == b"DATA_REPLY":
                _, _, kw, src, fid, data = msg
                self.n_arrived += 1
                if self.q.full():           # 최신 것만 남긴다
                    try:
                        self.q.get_nowait()
                        self.n_dropped += 1
                    except Empty:
                        pass
                self.q.put(data)

    def get_latest(self, timeout: float = 1.0) -> bytes | None:
        """가장 최근에 도착한 본문. 없으면 None."""
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            return None

    def send(self, payload: bytes, kw: bytes | None = None) -> None:
        """업로드한다. kw 를 주면 그 keyword 로, 아니면 result_kw 로 간다.

        UPLOAD 프레임이 keyword 를 실어 나르므로 소켓 하나로 여러 채널에 올릴 수 있다.
        지표(USER_STATE)와 기기 제어(HL2_CONTROL)처럼 방향이 같고 성격만 다른 경우에 쓴다.
        """
        if self.tx is None:
            raise RuntimeError("result_kw 없이 만든 클라이언트는 송신할 수 없다")
        self._fid += 1
        self.tx.send_multipart([b"", b"UPLOAD", kw or self.result_kw, self.identity,
                                str(self._fid).encode(), payload])

    def close(self) -> None:
        self._stop = True
        self.rx.close()
        if self.tx is not None:
            self.tx.close()
        self.ctx.term()


def add_broker_args(parser) -> None:
    """진입점마다 같은 이름으로 브로커 주소를 받게 한다."""
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"comm_hub 브로커 주소 (기본 {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"comm_hub 브로커 포트 (기본 {DEFAULT_PORT})")
