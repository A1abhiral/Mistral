import json
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.models import User, Question
from api.dependencies import get_current_user, get_db

router = APIRouter(prefix="/teacher", tags=["Teacher"])

@router.get("/questions")
async def get_student_questions(
    days: int = 7,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all student questions from the last N days (teacher only)"""
    
    # Verify user is a teacher
    if current_user.role != "teacher":
        raise HTTPException(status_code=403, detail="Only teachers can access this endpoint")
    
    # Calculate date threshold
    threshold_date = datetime.utcnow() - timedelta(days=days)
    
    # Query questions
    questions = db.query(Question).filter(
        Question.timestamp >= threshold_date
    ).order_by(Question.timestamp.desc()).all()
    
    # Format response
    response_questions = []
    for q in questions:
        student = db.query(User).filter(User.id == q.student_id).first()
        response_questions.append({
            "id": q.id,
            "student_id": q.student_id,
            "student_name": student.name if student else "Unknown",
            "question": q.question,
            "answer": q.answer,
            "retrieved": json.loads(q.retrieved_docs) if q.retrieved_docs else [],
            "timestamp": q.timestamp.isoformat()
        })
    
    return {"questions": response_questions}
