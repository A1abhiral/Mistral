import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas.schemas import SaveQuestionRequest
from db.models import User, Question
from api.dependencies import get_current_user, get_db

router = APIRouter(prefix="/student", tags=["Student"])

@router.post("/save-question")
async def save_question(
    request: SaveQuestionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save student question and answer to database"""
    
    # Verify user is a student
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can save questions")
    
    # Verify user_id matches current user
    if request.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot save questions for other users")
    
    # Create question record
    question = Question(
        student_id=current_user.id,
        question=request.question,
        answer=request.answer,
        retrieved_docs=json.dumps(request.retrieved),
        timestamp=datetime.utcnow()
    )
    
    db.add(question)
    db.commit()
    db.refresh(question)
    
    return {"success": True, "question_id": question.id}

@router.get("/history")
async def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all questions asked by the currently logged-in student"""
    
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can access history")
    
    questions = db.query(Question).filter(
        Question.student_id == current_user.id
    ).order_by(Question.timestamp.desc()).all()
    
    return {
        "questions": [
            {
                "id": q.id,
                "question": q.question,
                "answer": q.answer,
                "retrieved": json.loads(q.retrieved_docs) if q.retrieved_docs else [],
                "timestamp": q.timestamp.isoformat() if q.timestamp else None
            }
            for q in questions
        ]
    }
