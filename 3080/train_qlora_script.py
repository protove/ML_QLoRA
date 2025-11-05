#!/usr/bin/env python3
"""
QLoRA Fine-tuning Script for RTX 3080
Mistral-7B 모델을 RTX 3080에서 효율적으로 학습하기

이 스크립트는 백그라운드에서 실행되며 모든 로그를 파일에 기록합니다.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import gc
import time

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 그래프 저장
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class ExperimentManager:
    """실험 폴더 및 로깅 관리"""
    
    def __init__(self, base_dir="./qlora-mistral-3080-final"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.experiment_dir = self._create_experiment_dir()
        self.logger = self._setup_logging()
        
    def _create_experiment_dir(self):
        """다음 실험 번호의 폴더 생성 (0001, 0002, ...)"""
        existing_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        
        if existing_dirs:
            max_num = max([int(d.name) for d in existing_dirs])
            next_num = max_num + 1
        else:
            next_num = 1
        
        experiment_dir = self.base_dir / f"{next_num:04d}"
        experiment_dir.mkdir(exist_ok=True)
        
        return experiment_dir
    
    def _setup_logging(self):
        """로깅 설정"""
        log_file = self.experiment_dir / "training.log"
        
        # 로거 생성
        logger = logging.getLogger("QLoRA_Training")
        logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def save_config(self, bnb_config, lora_config, training_args):
        """설정 파일들을 보기 좋게 저장"""
        config_file = self.experiment_dir / "config.json"
        
        config_data = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "experiment_dir": str(self.experiment_dir),
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            },
            "BitsAndBytesConfig": {
                "load_in_4bit": bnb_config.load_in_4bit,
                "bnb_4bit_quant_type": bnb_config.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": str(bnb_config.bnb_4bit_compute_dtype),
                "bnb_4bit_use_double_quant": bnb_config.bnb_4bit_use_double_quant,
            },
            "LoraConfig": {
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "target_modules": lora_config.target_modules,
                "lora_dropout": lora_config.lora_dropout,
                "bias": lora_config.bias,
                "task_type": lora_config.task_type,
            },
            "TrainingArguments": {
                "output_dir": training_args.output_dir,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
                "optim": training_args.optim,
                "learning_rate": training_args.learning_rate,
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "num_train_epochs": training_args.num_train_epochs,
                "fp16": training_args.fp16,
                "logging_steps": training_args.logging_steps,
                "save_strategy": training_args.save_strategy,
                "gradient_checkpointing": training_args.gradient_checkpointing,
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # 텍스트 버전도 저장 (읽기 쉽게)
        config_txt_file = self.experiment_dir / "config.txt"
        with open(config_txt_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("QLoRA Training Configuration\n")
            f.write("="*80 + "\n\n")
            
            for section, values in config_data.items():
                f.write(f"\n[{section}]\n")
                f.write("-"*80 + "\n")
                for key, value in values.items():
                    f.write(f"{key:30s}: {value}\n")
        
        self.logger.info(f"✓ 설정 파일 저장 완료: {config_file}")
        self.logger.info(f"✓ 설정 파일 저장 완료: {config_txt_file}")


class LoggingCallback(TrainerCallback):
    """학습 과정을 로그 파일에 기록하는 콜백"""
    
    def __init__(self, logger, experiment_dir):
        self.logger = logger
        self.experiment_dir = Path(experiment_dir)
        self.losses = []
        self.steps = []
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.logger.info("="*80)
        self.logger.info("🚀 학습 시작!")
        self.logger.info("="*80)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.losses.append(logs['loss'])
                self.steps.append(state.global_step)
            
            # 로그 메시지 포맷팅
            log_msg = f"Step {state.global_step}/{state.max_steps} | "
            
            if 'loss' in logs:
                log_msg += f"Loss: {logs['loss']:.4f} | "
            if 'learning_rate' in logs:
                log_msg += f"LR: {logs['learning_rate']:.2e} | "
            if 'epoch' in logs:
                log_msg += f"Epoch: {logs['epoch']:.2f} | "
            
            # GPU 메모리 정보
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                log_msg += f"GPU: {gpu_mem:.2f}GB"
            
            self.logger.info(log_msg)
            
    def on_epoch_end(self, args, state, control, **kwargs):
        self.logger.info(f"✓ Epoch {state.epoch} 완료")
        if self.losses:
            recent_avg = np.mean(self.losses[-10:]) if len(self.losses) >= 10 else np.mean(self.losses)
            self.logger.info(f"  최근 평균 Loss: {recent_avg:.4f}")
        
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        
        self.logger.info("="*80)
        self.logger.info("✅ 학습 완료!")
        self.logger.info("="*80)
        self.logger.info(f"총 소요 시간: {total_time/60:.1f}분 ({total_time/3600:.2f}시간)")
        
        if self.losses:
            self.logger.info(f"시작 Loss: {self.losses[0]:.4f}")
            self.logger.info(f"최종 Loss: {self.losses[-1]:.4f}")
            self.logger.info(f"최소 Loss: {min(self.losses):.4f}")
            self.logger.info(f"평균 Loss: {np.mean(self.losses):.4f}")
            loss_reduction = ((self.losses[0] - self.losses[-1]) / self.losses[0] * 100)
            self.logger.info(f"Loss 감소율: {loss_reduction:.2f}%")
        
        # 최종 Loss 그래프 저장
        self.save_loss_plot()
        
    def save_loss_plot(self):
        """Loss 그래프를 파일로 저장"""
        if not self.losses:
            self.logger.warning("저장할 학습 데이터가 없습니다.")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plot_file = self.experiment_dir / "training_loss.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ 학습 곡선 저장 완료: {plot_file}")
        
        # Loss 데이터도 CSV로 저장
        loss_data_file = self.experiment_dir / "training_loss.csv"
        with open(loss_data_file, 'w') as f:
            f.write("step,loss\n")
            for step, loss in zip(self.steps, self.losses):
                f.write(f"{step},{loss}\n")
        
        self.logger.info(f"✓ Loss 데이터 저장 완료: {loss_data_file}")


def format_instruction(sample):
    """프롬프트 형식화 함수"""
    return f"""### Instruction:
{sample['instruction']}

### Context:
{sample['context']}

### Response:
{sample['response']}
"""


def data_collator(data):
    """데이터 콜레이터"""
    return {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
        'labels': torch.stack([torch.tensor(f['input_ids']) for f in data])
    }


def main():
    """메인 학습 함수"""
    
    # 실험 관리자 초기화
    exp_manager = ExperimentManager()
    logger = exp_manager.logger
    
    logger.info("="*80)
    logger.info("QLoRA Fine-tuning for RTX 3080")
    logger.info("Mistral-7B 모델 학습 스크립트")
    logger.info("="*80)
    logger.info(f"실험 폴더: {exp_manager.experiment_dir}")
    logger.info(f"PyTorch 버전: {torch.__version__}")
    logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA 버전: {torch.version.cuda}")
    else:
        logger.error("CUDA를 사용할 수 없습니다!")
        return
    
    # 1. 모델 ID 및 토크나이저 설정
    logger.info("\n" + "-"*80)
    logger.info("1. 토크나이저 로드 중...")
    logger.info("-"*80)
    
    model_id = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✓ 토크나이저 로드 완료")
    logger.info(f"  모델: {model_id}")
    logger.info(f"  Vocabulary 크기: {len(tokenizer)}")
    
    # 2. QLoRA 설정
    logger.info("\n" + "-"*80)
    logger.info("2. QLoRA 설정 중...")
    logger.info("-"*80)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    logger.info("✓ QLoRA 설정 완료")
    logger.info("  - 4비트 양자화: 활성화")
    logger.info("  - 양자화 타입: NF4")
    logger.info("  - 계산 타입: bfloat16")
    logger.info("  - 이중 양자화: 활성화")
    
    # 3. 모델 로드
    logger.info("\n" + "-"*80)
    logger.info("3. 모델 로드 중... (시간이 걸릴 수 있습니다)")
    logger.info("-"*80)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    logger.info("✓ 모델 로드 완료")
    logger.info(f"  모델 타입: {type(model).__name__}")
    logger.info(f"  디바이스: {next(model.parameters()).device}")
    
    # 4. LoRA 설정
    logger.info("\n" + "-"*80)
    logger.info("4. LoRA 설정 중...")
    logger.info("-"*80)
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    logger.info("✓ LoRA 설정 완료")
    logger.info(f"  - 랭크(r): {peft_config.r}")
    logger.info(f"  - 알파: {peft_config.lora_alpha}")
    logger.info(f"  - 드롭아웃: {peft_config.lora_dropout}")
    logger.info(f"  - 대상 모듈 수: {len(peft_config.target_modules)}")
    
    # 5. PEFT 모델 준비
    logger.info("\n" + "-"*80)
    logger.info("5. PEFT 모델 준비 중...")
    logger.info("-"*80)
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params
    
    logger.info("✓ PEFT 모델 준비 완료")
    logger.info(f"  학습 가능한 파라미터: {trainable_params:,}")
    logger.info(f"  전체 파라미터: {all_params:,}")
    logger.info(f"  학습 가능 비율: {trainable_percent:.4f}%")
    
    if torch.cuda.is_available():
        logger.info(f"  GPU 메모리 할당: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"  GPU 메모리 예약: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # 6. 데이터셋 로드 및 전처리
    logger.info("\n" + "-"*80)
    logger.info("6. 데이터셋 로드 중...")
    logger.info("-"*80)
    
    data = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle()
    
    logger.info(f"✓ 데이터셋 로드 완료")
    logger.info(f"  - 샘플 수: {len(data)}")
    logger.info(f"  - 컬럼: {data.column_names}")
    
    # 토크나이징
    logger.info("토크나이징 중...")
    tokenized_data = data.map(
        lambda p: tokenizer(format_instruction(p), truncation=True, max_length=512, padding="max_length"),
        remove_columns=data.column_names
    )
    
    logger.info("✓ 토크나이징 완료")
    logger.info(f"  - 최대 길이: 512 토큰")
    
    # 7. TrainingArguments 설정
    logger.info("\n" + "-"*80)
    logger.info("7. Training Arguments 설정 중...")
    logger.info("-"*80)
    
    training_args = TrainingArguments(
        output_dir=str(exp_manager.experiment_dir / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=5,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
    )
    
    logger.info("✓ Training Arguments 설정 완료")
    logger.info(f"  - 배치 크기: {training_args.per_device_train_batch_size}")
    logger.info(f"  - Gradient Accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  - 실질 배치 크기: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  - 학습률: {training_args.learning_rate}")
    logger.info(f"  - 스케줄러: {training_args.lr_scheduler_type}")
    logger.info(f"  - 에포크: {training_args.num_train_epochs}")
    
    # 설정 저장
    exp_manager.save_config(bnb_config, peft_config, training_args)
    
    # 8. Trainer 초기화
    logger.info("\n" + "-"*80)
    logger.info("8. Trainer 초기화 중...")
    logger.info("-"*80)
    
    logging_callback = LoggingCallback(logger, exp_manager.experiment_dir)
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_data,
        args=training_args,
        data_collator=data_collator,
        callbacks=[logging_callback]
    )
    
    logger.info("✓ Trainer 초기화 완료")
    
    # 9. 학습 시작
    logger.info("\n" + "="*80)
    logger.info("9. 학습 시작 🚀")
    logger.info("="*80 + "\n")
    
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}", exc_info=True)
        return
    
    # 10. 모델 저장
    logger.info("\n" + "-"*80)
    logger.info("10. 모델 저장 중...")
    logger.info("-"*80)
    
    model_save_dir = exp_manager.experiment_dir / "model"
    model_save_dir.mkdir(exist_ok=True)
    
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    
    logger.info(f"✓ 모델 저장 완료: {model_save_dir}")
    
    # 11. 최종 요약
    logger.info("\n" + "="*80)
    logger.info("📊 최종 학습 결과 요약")
    logger.info("="*80)
    
    if logging_callback.losses:
        logger.info(f"총 Step 수: {len(logging_callback.steps)}")
        logger.info(f"시작 Loss: {logging_callback.losses[0]:.4f}")
        logger.info(f"최종 Loss: {logging_callback.losses[-1]:.4f}")
        logger.info(f"최소 Loss: {min(logging_callback.losses):.4f}")
        logger.info(f"평균 Loss: {np.mean(logging_callback.losses):.4f}")
        loss_reduction = ((logging_callback.losses[0] - logging_callback.losses[-1]) / logging_callback.losses[0] * 100)
        logger.info(f"Loss 감소율: {loss_reduction:.2f}%")
    
    # 12. GPU 메모리 정리
    logger.info("\n" + "-"*80)
    logger.info("GPU 메모리 정리 중...")
    logger.info("-"*80)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"✓ GPU 메모리 정리 완료")
        logger.info(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    logger.info("\n" + "="*80)
    logger.info("🎉 모든 작업이 완료되었습니다!")
    logger.info(f"결과 저장 위치: {exp_manager.experiment_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
