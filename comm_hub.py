import asyncio
import zmq.asyncio
import logging
import argparse

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CentralHub")

class RobustCentralHub:
    def __init__(self, port: int):
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
        #스토리지 초기화
        self.storage = {}

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
            self.storage[cache_key] = data

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
            cache_key = f"{kw.decode()}_{source}_{fid.decode()}"

            if cache_key in self.storage:
                raw_data = self.storage[cache_key]

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
    args = parser.parse_args()

    hub = RobustCentralHub(port=args.port)
    try:
        asyncio.run(hub.run())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")

