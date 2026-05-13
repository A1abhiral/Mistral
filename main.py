from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import config
from db.database import engine, Base, SessionLocal
from db.models import User
from core.security import hash_password
from services.llm_service import load_model

from api.routes import admin, auth, chat, health, student, teacher

def seed_database():
    """Create admin and demo accounts on startup"""
    db = SessionLocal()
    
    try:
        # --- Always seed admin account ---
        admin_exists = db.query(User).filter(User.role == "admin").first()
        if not admin_exists:
            print(" Creating admin account...")
            admin_user = User(
                email=config.ADMIN_EMAIL,
                name=config.ADMIN_NAME,
                password_hash=hash_password(config.ADMIN_PASSWORD),
                role="admin"
            )
            db.add(admin_user)
            db.commit()
            print(f" Admin account created!")
            print(f"   Admin: {config.ADMIN_EMAIL} / {config.ADMIN_PASSWORD}")
        
        # --- Optionally seed demo accounts ---
        if not config.ENABLE_DEMO_ACCOUNTS:
            print("ℹ Demo accounts disabled")
            return
        
        # Check if demo users already exist
        demo_exists = db.query(User).filter(User.email == "student@demo.com").first()
        if demo_exists:
            return
        
        print(" Creating demo accounts...")
        
        # Create demo student
        student_user = User(
            email="student@demo.com",
            name="Alex Student",
            password_hash=hash_password("student123"),
            role="student"
        )
        
        # Create demo teacher
        teacher_user = User(
            email="teacher@demo.com",
            name="Prof. Smith",
            password_hash=hash_password("teacher123"),
            role="teacher"
        )
        
        db.add(student_user)
        db.add(teacher_user)
        db.commit()
        
        print(" Demo accounts created!")
        print("   Student: student@demo.com / student123")
        print("   Teacher: teacher@demo.com / teacher123")
    except Exception as e:
        print(f" Failed to seed database: {e}")
        db.rollback()
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    print("=" * 50)
    print("AI Tutor Backend Starting...")
    print("=" * 50)

    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Seed database
    seed_database()
    
    # Load AI model
    load_model()
    
    yield
    
    # Shutdown tasks (if any)
    print(" AI Tutor Backend Shutting Down...")

# App initialization
app = FastAPI(
    title="AI Tutor API",
    description="Backend API for AI Tutoring with LLM and RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(student.router)
app.include_router(teacher.router)
app.include_router(admin.router)

if __name__ == "__main__":
    import uvicorn
    
    print(f"\n Starting server on http://{config.HOST}:{config.PORT}")
    print(f"API docs available at http://{config.HOST}:{config.PORT}/docs\n")
    
    uvicorn.run(
        "main:app", 
        host=config.HOST, 
        port=config.PORT,
        log_level="info",
        reload=True
    )
