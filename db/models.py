from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base

class User(Base):
    """User model for students, teachers, and admins"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'student', 'teacher', or 'admin'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    questions = relationship("Question", back_populates="student", cascade="all, delete-orphan")


class Question(Base):
    """Question model for storing student queries and AI responses"""
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    retrieved_docs = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    student = relationship("User", back_populates="questions")
