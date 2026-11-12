#!/usr/bin/env bash
# HQS 상태(HL2_MODE)·패널(HL2_UI), 손 pose·gesture(SERVER_RESULT), 와치·폰 bridge 를 한 번에 띄운다.
# comm_hub.py 는 따로 먼저 띄워 둔다. Ctrl+C 한 번이면 셋 다 끄고, 하나가 끝나면 나머지도 끈다.
#   ./run_wiseui.sh
set -m          # 프로세스마다 process group 을 따로 둔다. Ctrl+C 는 이 스크립트만 받고 stop 이 나눠 준다
cd "$(dirname "$0")"
PY=~/anaconda3/envs/uvr_integ/bin/python

start() {       # start 이름표 스크립트 [인자...]   출력 줄마다 [이름표] 를 붙인다
    local tag=$1; shift
    "$PY" -u "$@" 2>&1 | (trap '' INT; exec sed -u "s/^/[$tag] /") &
}

stop() {
    trap '' INT TERM; trap - EXIT
    local pids; pids=$(jobs -p)         # 각 job 의 첫 프로세스, 곧 python
    # SIGINT(KeyboardInterrupt)로 끈다. 프로파일 저장·HL2 마이크 해제가 그 정리 코드에 있다
    kill -INT $pids 2>/dev/null
    for _ in $(seq 100); do             # 10 s 안에 안 끝나면 강제로 끈다
        kill -0 $pids 2>/dev/null || break
        sleep 0.1
    done
    kill -KILL $pids 2>/dev/null
    wait
}
trap stop INT TERM EXIT

start RECV main_all_hl2_receiver.py --hqs
start HAND main_handtrack.py
start BRIDGE main_devicebridge.py
wait -n
