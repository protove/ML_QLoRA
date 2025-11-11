import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig는 더 이상 필요하지 않습니다.
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
# prepare_model_for_kbit_training은 더 이상 필요하지 않습니다.

# 1. 모델 ID 및 토크나이저
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. QLoRA 설정 (BitsAndBytesConfig) -> 제거됨
# bnb_config = BitsAndBytesConfig(...) -> 이 부분을 삭제합니다.

# 3. 모델 로드 (FP16으로 직접 로드)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=bnb_config, -> 삭제
    torch_dtype=torch.float16,     # ◀◀◀ 4비트 대신 16비트로 직접 로드
    device_map="auto"
)

# 4. LoRA Config (PEFT) - QLoRA와 동일
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[ # TinyLlama (Llama 2 아키텍처)
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. PEFT 모델 준비
# model = prepare_model_for_kbit_training(model) -> k-bit 학습이 아니므로 삭제
model = get_peft_model(model, peft_config) # ◀◀◀ LoRA 어댑터만 적용

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()

# 6. 데이터셋 로드 및 전처리 (Dolly-15k 전체)
data = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle()

def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Context:
{sample['context']}

### Response:
{sample['response']}
"""

tokenized_data = data.map(lambda p: tokenizer(format_instruction(p), truncation=True, max_length=512, padding="max_length"))

# 7. TrainingArguments 설정 (VRAM 최적화 유지)
training_args = TrainingArguments(
    output_dir="./fp16-lora-tinyllama-3080", # ◀◀◀ 출력 폴더 변경
    per_device_train_batch_size=1,     # VRAM 한계로 1 유지 (OOM 발생 시)
    gradient_accumulation_steps=16,    # 실질 배치 크기 16 유지
    
    optim="paged_adamw_32bit",         # ◀◀◀ VRAM OOM 방지를 위해 그대로 사용
    
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    
    fp16=True,                         # ◀◀◀ FP16 학습 활성화 (필수)
    
    logging_steps=10,
    save_strategy="epoch",
    
    gradient_checkpointing=True,       # ◀◀◀ VRAM 절약을 위해 그대로 사용
)

# 8. Trainer 초기화 및 학습 시작
trainer = Trainer(
    model=model,
    train_dataset=tokenized_data,
    args=training_args,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                               'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                               'labels': torch.stack([f['input_ids'] for f in data])}
)

print("RTX 3080 최적화 'FP16 LoRA' 학습 시작...")
trainer.train()

print("학습 완료. 모델 저장 중...")
model.save_pretrained("./fp16-lora-tinyllama-3080-final")
tokenizer.save_pretrained("./fp16-lora-tinyllama-3080-final")