from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.models import User
from core.security import hash_password
from api.dependencies import get_db, require_admin
from schemas.schemas import CreateUserRequest, UserResponse, DeleteUserResponse

router = APIRouter(prefix="/admin", tags=["Admin"])

VALID_ROLES = {"student", "teacher", "admin"}


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """List all user accounts (admin only)"""
    users = db.query(User).order_by(User.created_at.desc()).all()
    
    return [
        UserResponse(
            id=u.id,
            email=u.email,
            name=u.name,
            role=u.role,
            created_at=u.created_at.isoformat() if u.created_at else None
        )
        for u in users
    ]


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    request: CreateUserRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create a new user account (admin only)"""
    
    # Validate role
    if request.role not in VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role '{request.role}'. Must be one of: {', '.join(VALID_ROLES)}"
        )
    
    # Check for duplicate email
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=409,
            detail=f"A user with email '{request.email}' already exists"
        )
    
    # Create the new user
    new_user = User(
        email=request.email,
        name=request.name,
        password_hash=hash_password(request.password),
        role=request.role
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        name=new_user.name,
        role=new_user.role,
        created_at=new_user.created_at.isoformat() if new_user.created_at else None
    )


@router.delete("/users/{user_id}", response_model=DeleteUserResponse)
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete a user account (admin only)"""
    
    # Prevent admin from deleting their own account
    if user_id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot delete your own admin account"
        )
    
    # Find the user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User with ID {user_id} not found"
        )
    
    user_email = user.email
    db.delete(user)  # Cascading delete removes associated questions
    db.commit()
    
    return DeleteUserResponse(
        success=True,
        message=f"User '{user_email}' (ID: {user_id}) has been deleted"
    )
