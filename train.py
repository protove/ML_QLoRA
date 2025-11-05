import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -----------------------------
# 0. MPS(Apple Silicon GPU) 확인
# -----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Training Device: {device}")

# -----------------------------
# 1. 모델 & 토크나이저 로드 (TinyLlama 1.1B)
# -----------------------------
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("📌 Loading TinyLlama model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# MPS에서는 bitsandbytes 사용 불가 → FP16 + MPS 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map={"": device}
)

# -----------------------------
# 2. LoRA 설정
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # TinyLlama 호환 모듈만 적용
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# k-bit 훈련 준비 (Mac에서도 필요)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print("✅ PEFT + LoRA Ready!")
model.print_trainable_parameters()

# -----------------------------
# 3. 데이터 로드 (Mini Alpaca 3k)
# -----------------------------
print("📂 Loading dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split="train[:3000]")  # 3k

def format_prompt(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Output:
{sample['output']}
"""

def tokenize(example):
    text = format_prompt(example)
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=False)

# -----------------------------
# 4. TrainingArguments 설정
# -----------------------------
training_args = TrainingArguments(
    output_dir="./tinyllama_lora_output_v2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,

    fp16=False,     # ❗ MPS는 fp16 대신 False 또는 bf16 사용
    bf16=False,     # MPS에서 bf16 미지원 → False 유지

    logging_dir="./logs",      # ✅ 로그 저장 폴더
    logging_steps=10,          # ✅ 10 step마다 loss 기록
    save_steps=375,            # ✅ 체크포인트 저장
    save_total_limit=3,        # ✅ 최근 3개만 저장
    report_to=["tensorboard"], # ✅ 텐서보드 로그 기록

    optim="adamw_torch",          # ⭐ MPS에서 가장 안정적
    lr_scheduler_type="cosine",   # ⭐ loss 안정화
    warmup_ratio=0.03,            # ⭐ 초반 loss 폭주 방지
)

# -----------------------------
# 5. Trainer 실행
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("🚀 Training Started...")
trainer.train()

# -----------------------------
# 6. 학습된 LoRA 어댑터 저장
# -----------------------------
save_path = "./tinyllama-lora-mac"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"🎉 Training Completed! LoRA adapter saved at: {save_path}")
