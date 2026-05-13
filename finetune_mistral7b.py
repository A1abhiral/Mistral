import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

print("CUDA:", torch.cuda.is_available())

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# 🔹 Quantization config (REQUIRED)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# 🔹 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 🔹 Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)


# 🔹 LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# 🔹 Dataset
train_dataset = load_dataset("json", data_files="train.jsonl")["train"]
val_dataset = load_dataset("json", data_files="val.jsonl")["train"]

def tokenize(examples):
    texts = []
    for msgs in examples["messages"]:
        text = ""
        for m in msgs:
            text += f"{m['role']}: {m['content']}\n"
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["messages"])
val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["messages"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# 🔹 Training args
training_args = TrainingArguments(
    output_dir="./mistral_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    eval_strategy="steps",   # ✅ NEW NAME
    eval_steps=500,
    save_total_limit=2,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
trainer.save_model("./mistral_lora_final")

