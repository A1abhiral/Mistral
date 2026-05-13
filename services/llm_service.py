import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from core.config import config

# Initialize as global variables to be loaded once
model = None
tokenizer = None

def load_model():
    """Load the LLM and LoRA adapter"""
    global model, tokenizer
    
    print("=" * 50)
    print(f"\n📦 Loading AI model...")
    print(f"   Base model: {config.BASE_MODEL}")
    print(f"   LoRA path: {config.LORA_PATH}")
    
    try:
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        print("   Loading base model (4-bit)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL,
            device_map="auto",
            quantization_config=bnb_config
        )
        
        print("   Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, config.LORA_PATH)
        model.eval()
        
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
        tokenizer = None
    
    print("=" * 50)

def get_model_status() -> bool:
    """Return True if model is successfully loaded"""
    return model is not None and tokenizer is not None

def generate_response(prompt_text: str) -> str:
    """Generate response from the loaded model"""
    if not get_model_status():
        raise ValueError("Model not available. Please check server logs.")
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove prompt echo
    if decoded.startswith(prompt_text):
        decoded = decoded[len(prompt_text):].strip()

    return decoded
