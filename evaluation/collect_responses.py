"""
==============================================================
  STEP 1: Collect Responses from Base Model & Fine-Tuned Model
==============================================================
Run this script ON GPU to generate answers from both models
for all test questions.

Usage:
    python evaluation/collect_responses.py

Output:
    evaluation/results/model_responses.json
"""

import json
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Paths ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

QUESTIONS_PATH = os.path.join(SCRIPT_DIR, "comparison_questions.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "model_responses.json")

BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = os.path.join(PROJECT_DIR, "mistral_lora_final")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load Questions ─────────────────────────────────────────
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"📋 Loaded {len(questions)} test questions")

# ── Quantization Config ───────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ── Helper: Generate Response ─────────────────────────────
def generate_response(model, tokenizer, system_prompt, question, max_new_tokens=300):
    """Generate response from a model given a question."""
    prompt = f"system: {system_prompt}\nuser: {question}\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the prompt echo
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()

    # Clean up: take only the assistant's response
    # Stop at next "user:" or "system:" if model generates more
    for stop_token in ["\nuser:", "\nsystem:", "\n\nuser:", "\n\nsystem:"]:
        if stop_token in decoded:
            decoded = decoded[:decoded.index(stop_token)].strip()

    return decoded


# ══════════════════════════════════════════════════════════
#  PHASE 1: Fine-Tuned Model Responses
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 1: Loading FINE-TUNED model (Base + LoRA)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16
)

finetuned_model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False
)
finetuned_model.eval()

print("✅ Fine-tuned model loaded!")

# Generate fine-tuned responses
results = []
for i, q in enumerate(questions):
    print(f"\n[{i+1}/{len(questions)}] {q['subject']}: {q['question'][:60]}...")

    start_time = time.time()
    finetuned_response = generate_response(
        finetuned_model, tokenizer,
        q["system_prompt"], q["question"]
    )
    ft_time = time.time() - start_time

    print(f"  Fine-tuned ({ft_time:.1f}s): {finetuned_response[:100]}...")

    results.append({
        "id": q["id"],
        "subject": q["subject"],
        "question": q["question"],
        "textbook_answer": q["textbook_answer"],
        "finetuned_response": finetuned_response,
        "finetuned_time": round(ft_time, 2),
        "base_response": "",  # filled in phase 2
        "base_time": 0
    })

# Free GPU memory for base model
del finetuned_model
del base_model
torch.cuda.empty_cache()
print("\n🧹 Freed GPU memory")

# ══════════════════════════════════════════════════════════
#  PHASE 2: Base Model Responses (WITHOUT LoRA)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 2: Loading BASE model (no LoRA)")
print("=" * 60)

base_model_only = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16
)
base_model_only.eval()

print("✅ Base model loaded!")

for i, q in enumerate(questions):
    print(f"\n[{i+1}/{len(questions)}] {q['subject']}: {q['question'][:60]}...")

    start_time = time.time()
    base_response = generate_response(
        base_model_only, tokenizer,
        q["system_prompt"], q["question"]
    )
    base_time = time.time() - start_time

    print(f"  Base ({base_time:.1f}s): {base_response[:100]}...")

    results[i]["base_response"] = base_response
    results[i]["base_time"] = round(base_time, 2)

# Free memory
del base_model_only
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════
#  SAVE RESULTS
# ══════════════════════════════════════════════════════════
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ All responses saved to: {OUTPUT_PATH}")
print(f"   Total questions: {len(results)}")
print(f"\n📌 Next step: Run 'python evaluation/eval_base_vs_finetuned.py'")
