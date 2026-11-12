#!/usr/bin/env bash
# 폰과 HL2 한 대를 잇는 phone bridge 를 띄운다. comm_hub.py 는 따로 먼저 띄워 둔다. Ctrl+C 로 끈다.
#   ./run_wiseui_A_designer.sh                 # HoloLens2_A
#   ./run_wiseui_A_designer.sh HoloLens2_D     # 다른 기기. HL2 앱의 clientIdentity
cd "$(dirname "$0")"
exec ~/anaconda3/envs/uvr_integ/bin/python -u main_phonebridge.py --device "${1:-HoloLens2_A}"
