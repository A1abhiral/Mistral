import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration from environment variables"""
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_tutor.db")
    
    # JWT Authentication
    SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE-THIS-SECRET-KEY-IN-PRODUCTION")
    if SECRET_KEY == "CHANGE-THIS-SECRET-KEY-IN-PRODUCTION":
        print("WARNING: Using default SECRET_KEY. Set SECRET_KEY in .env file!")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "43200"))
    
    # Model Configuration
    BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    LORA_PATH = os.getenv("LORA_PATH", "mistral_lora_final")
    
    # CORS Origins (comma-separated list, or "*" to allow all)
    _origins_raw = os.getenv(
        "ALLOWED_ORIGINS", 
        "http://localhost:5173,http://127.0.0.1:5173"
    )
    ALLOWED_ORIGINS = ["*"] if _origins_raw.strip() == "*" else _origins_raw.split(",")
    
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Feature Flags
    ENABLE_DEMO_ACCOUNTS = os.getenv("ENABLE_DEMO_ACCOUNTS", "true").lower() == "true"
    
    # Admin Seed Account
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@aitutor.com")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
    ADMIN_NAME = os.getenv("ADMIN_NAME", "System Admin")

config = Config()
