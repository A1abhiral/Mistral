from fastapi import APIRouter, HTTPException
from schemas.schemas import Prompt, RetrieveRequest
from services.llm_service import generate_response, get_model_status
from Retrieval.retrieve import retrieve_book_content

router = APIRouter(tags=["AI"])

@router.post("/generate")
def generate(prompt: Prompt):
    """Generate AI response using fine-tuned Mistral model"""
    if not get_model_status():
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please check server logs."
        )
    
    try:
        response_text = generate_response(prompt.text)
        return {"response": response_text}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

@router.post("/retrieve")
def retrieve(payload: RetrieveRequest):
    """Retrieve relevant content from textbooks using RAG"""
    results = retrieve_book_content(
        query=payload.query,
        k=payload.k
    )
    return {"results": results}
