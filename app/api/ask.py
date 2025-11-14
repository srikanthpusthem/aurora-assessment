from fastapi import APIRouter, HTTPException
import logging
from app.models.request import AskRequest
from app.models.response import AskResponse
from app.services.qa_engine import answer_question

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    POST /api/ask endpoint to answer questions about users.
    
    Args:
        request: AskRequest containing the question
        
    Returns:
        AskResponse with the answer
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Validate request (Pydantic handles this automatically)
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Call QA engine
        answer = await answer_question(question)
        
        # Return response
        return AskResponse(answer=answer)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing question"
        )

