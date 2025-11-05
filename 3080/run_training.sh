#!/bin/bash
# QLoRA 학습 백그라운드 실행 스크립트

echo "========================================"
echo "QLoRA Training 백그라운드 실행"
echo "========================================"
echo ""

# Python 스크립트 경로
SCRIPT_PATH="./train_qlora_script.py"

# 백그라운드 로그 파일
BG_LOG="./training_background.log"

# 스크립트 실행 (백그라운드)
echo "학습을 백그라운드에서 시작합니다..."
echo "로그 파일: $BG_LOG"
echo ""

nohup python3 "$SCRIPT_PATH" > "$BG_LOG" 2>&1 &

# PID 저장
PID=$!
echo $PID > ./training.pid

echo "✓ 백그라운드 실행 시작!"
echo "  PID: $PID"
echo ""
echo "명령어 안내:"
echo "  - 로그 확인: tail -f $BG_LOG"
echo "  - 프로세스 확인: ps -p $PID"
echo "  - 프로세스 종료: kill $PID"
echo ""
echo "학습이 완료되면 qlora-mistral-3080-final/ 폴더에 결과가 저장됩니다."
echo "========================================"
