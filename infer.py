import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ✅ LoRA 모델 경로
MODEL_DIR = "./tinyllama-lora-mac"

def load_model():
    print("📌 Loading base model + LoRA adapter...")

    # 🔥 Base Model Load (MPS 적용)
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map={"": "mps"},
    )

    # 🔥 Merge LoRA Adapter
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print("✅ LoRA adapter loaded successfully!\n")
    return model, tokenizer


def generate_answer(model, tokenizer, user_input):
    # 📍 Alpaca-style formatting optional (but simple mode here)
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to("mps")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=200,
            do_sample=True,          # ✅ 샘플링 활성화
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,  # ✅ 따라쓰기 감소
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("🧠 TinyLlama + LoRA Inference Ready!")
    print("종료하려면 'quit' 입력\n")

    while True:
        user_input = input("💬 You: ")
        if user_input.lower() == "quit":
            print("\n👋 종료합니다. 수고했어!")
            break

        answer = generate_answer(model, tokenizer, user_input)
        print(f"🤖 Model: {answer}\n")
