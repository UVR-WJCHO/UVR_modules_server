import asyncio
import zmq.asyncio
import logging
import argparse
from collections import OrderedDict

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CentralHub")

# 스토리지 상한 (keyword 별로 따로 적용).
# UPLOAD 는 저장 후 알림을 보내고, 구독자는 곧바로 DOWNLOAD 로 받아간다. 즉 오래된
# 항목은 이미 받아갔거나 아무도 안 받아간 것이라 버려도 된다.
# keyword 별로 나눈 이유: 수 MB 짜리 MESH_RESULT 하나가 30Hz 로 흐르는 HL2DATA 를
# 밀어내 아직 다운로드 안 된 프레임을 지우는 일이 없어야 한다.
DEFAULT_CACHE_ITEMS = 64        # keyword 당 보관 개수
DEFAULT_CACHE_MB = 128          # keyword 당 보관 용량

class RobustCentralHub:
    def __init__(self, port: int, cache_items: int = DEFAULT_CACHE_ITEMS,
                 cache_mb: float = DEFAULT_CACHE_MB):
        self.port = port
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)

        # 포트 바인딩 (설정된 포트 사용)
        self.socket.bind(f"tcp://*:{self.port}")

        # 액션별 기대 프레임 수
        self.EXPECTED_FRAME_COUNTS = {
            b"SEND_REG": 5,  # [ID, Empty, Action, KW, Source] #삭제 
            b"RECV_REG": 6,  # [ID, Empty, Action, KW, Source, Target]
            b"UPLOAD": 7,  # [ID, Empty, Action, KW, Source, FID, Data]
            b"DOWNLOAD": 6,  # [ID, Empty, Action, KW, Source, FID]
            b"NOTIFY": 6,  # [ID, Empty, Action, KW, Source, FID]
            b"DATA_REPLY": 7,  # [ID, Empty, Action, KW, Source, FID, Data]
        }

        #리스너 초기화
        self.listeners = {}  # { "image": {id1, id2}, "seg_res": {id3} }
        self.src_to_id = {}  #{source : identity}
        self.source_to_kws = {}  # { source_b: {kw1, kw2, ...} }
        #스토리지 초기화 (keyword -> OrderedDict{cache_key: data}, 오래된 것이 앞)
        self.storage = {}
        self.cache_items = cache_items
        self.cache_bytes = int(cache_mb * 1024 * 1024)
        self.cache_size = {}      # keyword -> 현재 바이트 합
        self.evicted = {}         # keyword -> 버린 개수 (누적)

    def _store(self, kw_s: str, cache_key: str, data: bytes):
        """저장하고 상한을 넘은 만큼 오래된 것부터 버린다."""
        buf = self.storage.setdefault(kw_s, OrderedDict())
        old = buf.pop(cache_key, None)
        size = self.cache_size.get(kw_s, 0) - (len(old) if old is not None else 0)
        buf[cache_key] = data
        size += len(data)

        n_dropped = 0
        while buf and (len(buf) > self.cache_items or size > self.cache_bytes):
            _, dropped = buf.popitem(last=False)
            size -= len(dropped)
            n_dropped += 1
        self.cache_size[kw_s] = size

        if n_dropped:
            total = self.evicted.get(kw_s, 0) + n_dropped
            self.evicted[kw_s] = total
            # 매번 찍으면 30Hz 스트림에서 로그가 묻힌다. 처음과 이후 간헐적으로만.
            if total == n_dropped or total % 500 < n_dropped:
                logger.info(f"[EVICT] {kw_s}: dropped {total} total, "
                            f"holding {len(buf)} items / {size / 1024 / 1024:.1f}MB")

    async def run(self):
        logger.info(f"[*] Robust Central Hub started on PORT: {self.port}")
        while True:
            try:
                msg = await self.socket.recv_multipart()

                # 에러 감지 로직
                if len(msg) < 3:
                    logger.warning("Malformed packet: too short")
                    continue

                identity,_, action = msg[:3]

                if action not in self.EXPECTED_FRAME_COUNTS:
                    logger.error(f"Unknown action: {action} from {identity}")
                    continue
                if action == b"RECV_REG":
                    if len(msg) < 6 :
                        logger.error(
                            f"data mismatch for {action}: Expected {self.EXPECTED_FRAME_COUNTS[action]}, got {len(msg)}")
                        continue
                elif action == b"NOTIFY":
                    if len(msg) < 6:
                        logger.error(
                            f"data mismatch for {action}: Expected {self.EXPECTED_FRAME_COUNTS[action]}, got {len(msg)}")
                        continue
                elif len(msg) != self.EXPECTED_FRAME_COUNTS[action]:
                    logger.error(
                        f"data mismatch for {action}: Expected {self.EXPECTED_FRAME_COUNTS[action]}, got {len(msg)}")
                    continue

                await self.process_message(msg)

            except Exception as e:
                logger.error(f"Critical error: {e}")

    async def process_message(self, msg):

        identity, _, action = msg[:3]
        params = msg[3:]

        if action == b"RECV_REG":
            # 구조: [KW, Source, Target]
            keywords = params[:-2]
            source_b = params[-2]
            target_b = params[-1]

            # 1. 기존 등록 정보 효율적 삭제
            if source_b in self.source_to_kws:
                # 이 기기가 이전에 등록했던 키워드들만 순회
                for prev_kw in self.source_to_kws[source_b]:
                    if prev_kw in self.listeners:
                        # 해당 키워드 세트에서 본인(source_b) 항목만 필터링하여 제거
                        self.listeners[prev_kw] = {
                            item for item in self.listeners[prev_kw]
                            if item[0] != source_b
                        }
                # 추적 리스트 초기화
                self.source_to_kws[source_b] = set()
            else:
                self.source_to_kws[source_b] = set()

            # 2. 새로운 Identity 및 키워드 등록
            self.src_to_id[source_b] = identity

            # 해당 키워드의 리스너 집합(set)에 identity 추가
            for kw in keywords:
                kw_s = kw.decode()
                if kw_s not in self.listeners:
                    self.listeners[kw_s] = set()
                # 리스너 추가
                self.listeners[kw_s].add((source_b, target_b))
                # 역방향 인덱스에 추가 (나중에 삭제할 때 사용)
                self.source_to_kws[source_b].add(kw_s)
                logger.info(f"[REG] {source_b.decode()} registered for {kw_s}")

        # 2. 데이터 업로드 (UPLOAD)
        elif action == b"UPLOAD":
            # 구조: [KW, Source(frame sender), Target, FID, Data]
            kw, source, fid, data = params
            kw_s = kw.decode()
            #target_s = target.decode()
            #source_id = source  # 메시지를 보낸 기기의 실제 ZeroMQ ID, identity로 하면 전송자가 되는데 서버가 되기에 수정

            # 데이터 저장 (나중에 DOWNLOAD 할 수 있도록)
            cache_key = f"{kw_s}_{source}_{fid.decode()}"
            self._store(kw_s, cache_key, data)

            # 알림 전송 (Fan-out)
            if kw_s in self.listeners:
                notify_tasks = []
                for listener_src_b, listener_target_b in self.listeners[kw_s]:

                    # [구분 로직 시작]
                    listener_id = self.src_to_id[listener_src_b]
                    should_send = False
                    if listener_target_b == b"ALL":
                        # 1. 전부 받는 경우: 모든 리스너에게 전송
                        should_send = True
                    elif listener_target_b == b"MY_ONLY":
                        # 2. 자기 것만 받는 경우: 리스너 ID와 송신자 ID가 같을 때만 전송
                        if listener_src_b == source:
                            should_send = True
                    #elif listener_id == target:  # 특정 타겟 ID가 지정된 경우
                    #    should_send = True
                    # [구분 로직 끝]

                    # 알림 패킷: [ID, Empty, NOTIFY, KW, Source, Target, FID]
                    # 수신자가 한 명이라도 서버는 동시에 알림을 뿌립니다.
                    if should_send:
                        notify_tasks.append(
                            self.socket.send_multipart([
                                listener_id, b"", b"NOTIFY", kw, source, fid
                            ])
                        )

                if notify_tasks:
                    await asyncio.gather(*notify_tasks)
                    logger.info(f"[PUSH] Notified {len(notify_tasks)} nodes about {kw_s}")

        # 3. 데이터 다운로드 (DOWNLOAD)
        elif action == b"DOWNLOAD":
            # 구조: [KW, Source, Target, FID]
            kw, source, fid = params
            kw_s = kw.decode()
            cache_key = f"{kw_s}_{source}_{fid.decode()}"
            raw_data = self.storage.get(kw_s, {}).get(cache_key)

            if raw_data is not None:

                # 응답 패킷 구성: [요청자ID, Empty, DATA_REPLY, KW, Source, Target, FID, Data]
                # 여기서 Source와 Target은 원본 데이터를 보존하여 전달합니다.
                reply_msg = [
                    identity,  # 요청한 놈한테 다시 보내기 위해 ID 지정
                    b"",  # Empty Delimiter
                    b"DATA_REPLY",
                    kw,
                    source,
                    fid,
                    raw_data  # 찾아낸 바이너리 데이터
                ]
                await self.socket.send_multipart(reply_msg)
                logger.info(f"[SERVE] Sent {cache_key} to {identity}")
            else:
                logger.warning(f"[MISS] Data not found for key: {cache_key}")
        # 4. 프로세스 요청 처리
        elif action == b"NOTIFY":

            kw, source, fid, data = params
            kw_s = kw.decode()
            #target_s = target.decode()
            source_id = source
            #print('request method', kw, data)

            # 알림 전송 (Fan-out)
            # request method
            if kw_s in self.listeners:
                notify_tasks = []
                for listener_src_b, target_b in self.listeners[kw_s]:

                    #if target == listener_id:
                    #    should_send = True
                    listener_id = self.src_to_id[listener_src_b]
                    should_send = True

                    if should_send:
                        notify_tasks.append(
                            self.socket.send_multipart([
                                listener_id, b"", b"NOTIFY", kw, source, fid,data
                            ])
                        )

                if notify_tasks:
                    await asyncio.gather(*notify_tasks)
                    logger.info(f"[PUSH] Notified {len(notify_tasks)} nodes about {kw_s}")



if __name__ == "__main__":
    # 커맨드라인 인자 처리
    parser = argparse.ArgumentParser(description="ZeroMQ Central Hub Server")
    parser.add_argument("--port", type=int, default=37001, help="Port to bind the server (default: 5555)")
    parser.add_argument("--cache-items", type=int, default=DEFAULT_CACHE_ITEMS,
                        help=f"keyword 당 보관 개수 (default: {DEFAULT_CACHE_ITEMS})")
    parser.add_argument("--cache-mb", type=float, default=DEFAULT_CACHE_MB,
                        help=f"keyword 당 보관 용량 MB (default: {DEFAULT_CACHE_MB})")
    args = parser.parse_args()

    hub = RobustCentralHub(port=args.port, cache_items=args.cache_items,
                           cache_mb=args.cache_mb)
    try:
        asyncio.run(hub.run())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")

