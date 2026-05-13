from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

class Prompt(BaseModel):
    """Request schema for text generation"""
    text: str

class RetrieveRequest(BaseModel):
    """Request schema for document retrieval"""
    query: str
    k: int = 1

class LoginRequest(BaseModel):
    """Request schema for user login"""
    email: str
    password: str
    role: str

class LoginResponse(BaseModel):
    """Response schema for successful login"""
    user_id: int
    email: str
    name: str
    role: str
    token: str

class SaveQuestionRequest(BaseModel):
    """Request schema for saving student questions"""
    question: str
    answer: str
    retrieved: List[dict]
    user_id: int

class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model: str
    database: str

class CreateUserRequest(BaseModel):
    """Request schema for admin to create a new user account"""
    email: str
    name: str
    password: str
    role: str  # 'student', 'teacher', or 'admin'

class UserResponse(BaseModel):
    """Response schema for user details"""
    id: int
    email: str
    name: str
    role: str
    created_at: Optional[str] = None

class DeleteUserResponse(BaseModel):
    """Response schema for user deletion"""
    success: bool
    message: str
