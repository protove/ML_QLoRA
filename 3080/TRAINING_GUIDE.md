# QLoRA Training Script for RTX 3080

백그라운드에서 실행되고 로그를 자동으로 기록하는 QLoRA 학습 스크립트입니다.

## 📁 파일 구조

- `train_qlora_script.py` - 메인 학습 스크립트
- `run_training.sh` - 백그라운드 실행 스크립트
- `check_status.sh` - 학습 상태 확인 스크립트

## 🚀 사용법

### 1. 학습 시작 (백그라운드)

```bash
bash run_training.sh
```

또는

```bash
nohup python3 train_qlora_script.py > training_background.log 2>&1 &
```

### 2. 학습 상태 확인

```bash
bash check_status.sh
```

### 3. 실시간 로그 확인

```bash
# 백그라운드 로그
tail -f training_background.log

# 또는 최신 실험 폴더의 로그
tail -f qlora-mistral-3080-final/0001/training.log
```

### 4. 학습 중지

```bash
# PID 확인
cat training.pid

# 프로세스 종료
kill $(cat training.pid)
```

## 📊 결과 저장 구조

학습이 완료되면 `qlora-mistral-3080-final/` 폴더에 자동으로 번호가 매겨진 폴더가 생성됩니다:

```
qlora-mistral-3080-final/
├── 0001/                           # 첫 번째 실험
│   ├── config.json                 # 설정 정보 (JSON)
│   ├── config.txt                  # 설정 정보 (텍스트)
│   ├── training.log                # 학습 로그
│   ├── training_loss.png           # Loss 그래프
│   ├── training_loss.csv           # Loss 데이터
│   ├── checkpoints/                # 중간 체크포인트
│   │   └── checkpoint-xxx/
│   └── model/                      # 최종 학습 모델
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
├── 0002/                           # 두 번째 실험
│   └── ...
└── 0003/                           # 세 번째 실험
    └── ...
```

## 📝 저장되는 파일들

### 1. `config.json` / `config.txt`
학습에 사용된 모든 설정값이 저장됩니다:
- **BitsAndBytesConfig** - 양자화 설정
- **LoraConfig** - LoRA 파라미터
- **TrainingArguments** - 학습 하이퍼파라미터
- **실험 정보** - 타임스탬프, GPU 정보 등

### 2. `training.log`
학습 과정의 모든 로그:
- 각 스텝의 Loss
- Learning Rate 변화
- GPU 메모리 사용량
- 에포크별 진행 상황
- 최종 통계

### 3. `training_loss.png`
학습 Loss 곡선 그래프

### 4. `training_loss.csv`
각 스텝의 Loss 값 (CSV 형식)

### 5. `model/`
학습 완료된 LoRA 어댑터 모델

## 🔍 주요 기능

### ✅ 자동 폴더 번호 매기기
- 기존 폴더 확인 후 다음 번호로 자동 생성
- 0001, 0002, 0003... 형식

### ✅ 체계적인 로깅
- 파일과 콘솔 모두 로깅
- 타임스탬프 포함
- 학습 진행 상황 실시간 기록

### ✅ 설정 자동 저장
- JSON과 텍스트 두 가지 형식으로 저장
- 모든 하이퍼파라미터 기록
- 재현 가능성 보장

### ✅ 시각화 자동 저장
- Loss 그래프 PNG 파일로 저장
- CSV 데이터로도 제공

### ✅ GPU 메모리 최적화
- 4비트 양자화 (QLoRA)
- Gradient Checkpointing
- RTX 3080 최적화 설정

## ⚙️ 설정 커스터마이징

`train_qlora_script.py`의 `main()` 함수에서 다음 값들을 수정할 수 있습니다:

### LoRA 설정
```python
peft_config = LoraConfig(
    r=8,                    # 랭크 (↓ 메모리 절약, ↓ 성능)
    lora_alpha=16,          # 알파 (보통 r * 2)
    lora_dropout=0.05,      # 드롭아웃
    # ...
)
```

### 학습 설정
```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,      # 배치 크기
    gradient_accumulation_steps=16,     # Gradient 누적
    learning_rate=2e-4,                 # 학습률
    num_train_epochs=5,                 # 에포크 수
    # ...
)
```

## 📈 학습 모니터링

### 실시간 로그 확인
```bash
# 최신 실험 폴더의 로그 확인
tail -f qlora-mistral-3080-final/$(ls -t qlora-mistral-3080-final/ | head -1)/training.log
```

### GPU 사용량 확인
```bash
watch -n 1 nvidia-smi
```

### 프로세스 확인
```bash
ps aux | grep train_qlora_script.py
```

## 🛠️ 문제 해결

### CUDA Out of Memory
- `per_device_train_batch_size`를 1로 유지
- `gradient_accumulation_steps` 증가
- `r` (LoRA rank) 감소 (예: 8 → 4)

### 학습 속도 느림
- `gradient_checkpointing=False` (메모리 여유가 있다면)
- 데이터셋 크기 축소

### 프로세스가 백그라운드에서 종료됨
- `training_background.log` 확인
- GPU 드라이버 확인
- CUDA 메모리 부족 확인

## 📊 실험 비교

여러 실험의 결과를 비교하려면:

```bash
# 각 실험의 config.json 확인
cat qlora-mistral-3080-final/0001/config.json
cat qlora-mistral-3080-final/0002/config.json

# Loss 그래프 비교
eog qlora-mistral-3080-final/*/training_loss.png
```

## 💡 팁

1. **장시간 학습 시** SSH 세션이 끊어져도 계속 실행됩니다 (nohup 사용)
2. **여러 실험 동시 실행 가능** (GPU 메모리가 충분하다면)
3. **로그 파일로 학습 완료 확인** - "🎉 모든 작업이 완료되었습니다!" 메시지 확인
4. **실험 번호로 버전 관리** - Git에 코드만 올리고 모델은 로컬에 보관

## 📦 요구 사항

```bash
pip install torch transformers datasets peft bitsandbytes accelerate matplotlib
```

또는

```bash
pip install -r requirements_3080.txt
```

---

**Happy Training! 🚀**
