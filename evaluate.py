import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import math

# -----------------------------
# 1. 설정
# -----------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER_PATH = "./tinyllama-lora-mac"   # LoRA 저장된 경로
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"✅ Evaluation Device: {device}")


# -----------------------------
# 2. 모델 로드 함수
# -----------------------------
def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    return model, tokenizer

def load_lora_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    return model, tokenizer


# -----------------------------
# 3. Perplexity 계산 함수
# -----------------------------
def calculate_ppl(model, tokenizer, split="train[:200]"):
    dataset = load_dataset("tatsu-lab/alpaca", split=split)

    def tokenize(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Output:\n{example['output']}"
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            loss = model(**tokens, labels=tokens["input_ids"]).loss
        return float(loss)

    losses = [tokenize(example) for example in dataset]
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# -----------------------------
# 4. 두 모델 비교 실행
# -----------------------------
print("\n📍 Loading Models...")
base_model, base_tok = load_base_model()
lora_model, lora_tok = load_lora_model()

print("\n📊 Evaluating Base Model...")
base_loss, base_ppl = calculate_ppl(base_model, base_tok)
print(f"🔹 Base Model Loss: {base_loss:.4f} | PPL: {base_ppl:.2f}")

print("\n📊 Evaluating LoRA Model...")
lora_loss, lora_ppl = calculate_ppl(lora_model, lora_tok)
print(f"🔸 LoRA Model Loss: {lora_loss:.4f} | PPL: {lora_ppl:.2f}")

print("\n✅ Done! Performance Comparison:")
print("-----------------------------------")
print(f"Base PPL: {base_ppl:.2f}")
print(f"LoRA PPL: {lora_ppl:.2f}")
print("-----------------------------------")

# -----------------------------
# 5. 샘플 비교 테스트
# -----------------------------
test_prompt = "Explain what Reinforcement Learning is in one short paragraph."

print("\n🧪 Sample Output Comparison\n")

def generate(model, tok, name):
    tokens = tok(test_prompt, return_tensors="pt").to(device)
    output = model.generate(**tokens, max_new_tokens=100)
    print(f"----- {name} -----")
    print(tok.decode(output[0], skip_special_tokens=True))
    print()

generate(base_model, base_tok, "Base Model")
generate(lora_model, lora_tok, "LoRA Fine-Tuned Model")
