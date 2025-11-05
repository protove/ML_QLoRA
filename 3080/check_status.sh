#!/bin/bash
# QLoRA 학습 상태 확인 스크립트

echo "========================================"
echo "QLoRA Training 상태 확인"
echo "========================================"
echo ""

# PID 파일 확인
if [ -f "./training.pid" ]; then
    PID=$(cat ./training.pid)
    
    # 프로세스가 실행 중인지 확인
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ 학습이 실행 중입니다."
        echo "  PID: $PID"
        echo ""
        
        # CPU 및 메모리 사용량
        echo "리소스 사용량:"
        ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd --no-headers
        echo ""
        
        # GPU 사용량 (nvidia-smi가 있는 경우)
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU 사용량:"
            nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
            echo ""
        fi
        
        # 최신 실험 폴더 찾기
        LATEST_EXP=$(ls -d qlora-mistral-3080-final/[0-9][0-9][0-9][0-9] 2>/dev/null | sort -r | head -n 1)
        
        if [ -n "$LATEST_EXP" ]; then
            LOG_FILE="$LATEST_EXP/training.log"
            
            if [ -f "$LOG_FILE" ]; then
                echo "최신 로그 (마지막 20줄):"
                echo "----------------------------------------"
                tail -n 20 "$LOG_FILE"
                echo "----------------------------------------"
                echo ""
                echo "전체 로그 보기: tail -f $LOG_FILE"
            fi
        fi
        
    else
        echo "✗ 학습 프로세스를 찾을 수 없습니다."
        echo "  (PID $PID는 종료되었거나 존재하지 않습니다)"
        rm -f ./training.pid
    fi
else
    echo "✗ 실행 중인 학습이 없습니다."
    echo ""
    echo "학습을 시작하려면: bash run_training.sh"
fi

echo ""
echo "========================================"
