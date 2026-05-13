from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas.schemas import LoginRequest, LoginResponse
from db.models import User
from core.security import verify_password, create_access_token
from api.dependencies import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """User login endpoint"""
    user = db.query(User).filter(User.email == request.email).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if user.role != request.role:
        raise HTTPException(
            status_code=401, 
            detail=f"This account is not a {request.role} account"
        )
    
    token = create_access_token({"user_id": user.id, "role": user.role})
    
    return LoginResponse(
        user_id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        token=token
    )
