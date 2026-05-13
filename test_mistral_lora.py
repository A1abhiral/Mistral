import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("CUDA:", torch.cuda.is_available())

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
lora_path = "./mistral_lora_final"

# ✅ USE 4-BIT FOR INFERENCE (NO CPU OFFLOAD)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Load base model fully on GPU
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},   # FORCE GPU
    torch_dtype=torch.float16
)

# ✅ Load LoRA safely
model = PeftModel.from_pretrained(
    base_model,
    lora_path,
    is_trainable=False
)

model.eval()

prompt = """system: You are a helpful AI tutor.
user: Why is prism used to produce spectrum?
assistant:"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
